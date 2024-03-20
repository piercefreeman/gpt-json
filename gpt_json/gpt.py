import logging
from asyncio import TimeoutError as AsyncTimeoutError
from asyncio import wait_for
from copy import copy
from json import dumps as json_dumps
from json import loads as json_loads
from json.decoder import JSONDecodeError
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

import backoff
import openai
from openai.error import APIConnectionError, RateLimitError
from openai.error import Timeout as OpenAITimeout
from pydantic import BaseModel, Field, ValidationError
from tiktoken import encoding_for_model

from gpt_json.exceptions import InvalidFunctionParameters, InvalidFunctionResponse
from gpt_json.fn_calling import (
    function_to_name,
    get_argument_for_function,
    parse_function,
)
from gpt_json.models import (
    FixTransforms,
    GPTMessage,
    GPTModelVersion,
    ResponseType,
    TruncationOptions,
)
from gpt_json.parsers import find_json_response
from gpt_json.prompts import generate_schema_prompt
from gpt_json.streaming import (
    StreamingObject,
    parse_streamed_json,
    prepare_streaming_object,
)
from gpt_json.transformations import fix_bools, fix_truncated_json
from gpt_json.truncation import num_tokens_from_messages, truncate_tokens
from gpt_json.types_oai import ChatCompletionChunk

logger = logging.getLogger("gptjson_logger")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def handle_backoff(details):
    logger.warning(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with args {args} and kwargs "
        "{kwargs}".format(**details)
    )


SCHEMA_PROMPT_TEMPLATE_KEY = "json_schema"
SCHEMA_PROMPT_FUNCTION_KEY = "functions"


SchemaType = TypeVar("SchemaType", bound=BaseModel)


class ListResponse(BaseModel, Generic[SchemaType]):
    """
    Helper schema to echo back a list of input schemas
    """

    items: list[SchemaType] = Field(
        description="Repeat for as many objects as are relevant"
    )


class RunResponse(BaseModel, Generic[SchemaType]):
    """
    Helper schema to wrap a single response alongside the extracted metadata
    """

    raw_response: GPTMessage | None
    response: SchemaType | None
    fix_transforms: FixTransforms | None
    function_call: Callable[[BaseModel], Any] | None
    function_arg: BaseModel | None


class GPTJSON(Generic[SchemaType]):
    """
    A wrapper over GPT that provides basic JSON parsing and response handling.

    """

    _cls_schema_model: Type[SchemaType] | None = None
    schema_model: Type[SchemaType] | None = None

    def __init__(
        self,
        api_key: str | None = None,
        model: GPTModelVersion | str = GPTModelVersion.GPT_4,
        auto_trim: bool = False,
        auto_trim_response_overhead: int = 0,
        functions: list[Callable[[Any], Any]] | None = None,
        # For messages that are relatively deterministic
        temperature=0.2,
        timeout: int | None = None,
        openai_max_retries=3,
        **kwargs,
    ):
        """
        :param api_key: OpenAI API key, if `OPENAI_API_KEY` environment variable is not set
        :param model: GPTModelVersion or string model name
        :param auto_trim: If True, automatically trim messages to fit within the model's token limit
        :param auto_trim_response_overhead: If auto_trim is True, will leave at least `auto_trim_response_overhead` space
            for the output payload. For GPT, initial prompt + response <= allowed tokens.
        :param temperature: Temperature (or variation) of response payload; 0 is the most deterministic, 1 is the most random
        :param timeout: Timeout in seconds for each OpenAI API calls before timing out.
        :param openai_max_retries: Amount of times to retry failed API calls, caused by often transient load or rate limit issues
        :param kwargs: Additional arguments to pass to OpenAI's `openai.Completion.create` method

        """
        self.model = model.value if isinstance(model, GPTModelVersion) else model
        self.auto_trim = auto_trim
        self.temperature = temperature
        self.timeout = timeout
        self.openai_max_retries = openai_max_retries
        self.openai_arguments = kwargs
        self.schema_model = self._cls_schema_model
        self.functions = {
            function_to_name(cast_fn): cast_fn
            for fn in (functions or [])
            # Use an explicit cast; Callable can't be typehinted with BaseModel directly
            # because [BaseModel] is considered invariant with subclasses
            for cast_fn in [cast(Callable[[BaseModel], Any], fn)]
        }
        self.__class__._cls_schema_model = None

        if not self.schema_model:
            raise ValueError(
                "GPTJSON needs to be instantiated with a schema model, like GPTJSON[MySchema](...args)."
            )

        schema_origin = get_origin(self.schema_model)
        schema_args = get_args(self.schema_model)

        if (
            schema_origin
            and isinstance(schema_origin, type)
            and all(isinstance(arg, type) for arg in schema_args)
            and issubclass(schema_origin, BaseModel)
            and all(issubclass(arg, BaseModel) for arg in schema_args)
        ):
            self.schema_model = self.schema_model
            self.extract_type = ResponseType.DICTIONARY
        elif issubclass(self.schema_model, BaseModel):
            self.extract_type = ResponseType.DICTIONARY
        else:
            raise ValueError(
                "GPTJSON needs to be instantiated with a pydantic.BaseModel schema."
            )

        if "gpt-4" in self.model:
            self.max_tokens = (
                8192 - auto_trim_response_overhead if self.auto_trim else 8192
            )
        elif "gpt-3.5" in self.model:
            self.max_tokens = (
                4096 - auto_trim_response_overhead if self.auto_trim else 4096
            )
        else:
            raise ValueError(
                "Unknown model to infer max tokens, see https://platform.openai.com/docs/models/gpt-4 for more information on token length."
            )

        self.schema_prompt = generate_schema_prompt(self.schema_model)
        self.api_key = api_key

    async def run(
        self,
        messages: list[GPTMessage],
        max_response_tokens: int | None = None,
        format_variables: dict[str, Any] | None = None,
        truncation_options: TruncationOptions | None = None,
        allow_functions: bool = True,
    ) -> RunResponse[SchemaType]:
        """
        :param messages: List of GPTMessage objects to send to the API
        :param max_response_tokens: Maximum number of tokens allowed in the response
        :param format_variables: Variables to format into the message template. Uses standard
            Python string formatting, like "Hello {name}".format(name="World")

        :return: Tuple of (parsed response, fix transforms). The transformations here is a object that
            contains the allowed modifications that we might do to cleanup a GPT payload. It allows client callers
            to decide whether they want to allow the modifications or not.

        """
        messages = self.fill_messages(
            messages, format_variables, truncation_options, max_response_tokens
        )

        # Most requests succeed on the first try but we wrap it locally here in case
        # there is some temporarily instability with the API. If there are longer periods
        # of instability, there should be system-wide retries in a daemon.
        backoff_request_submission = backoff.on_exception(
            backoff.expo,
            (RateLimitError, OpenAITimeout, APIConnectionError),
            max_tries=self.openai_max_retries,
            on_backoff=handle_backoff,
        )(self.submit_request)

        response = await backoff_request_submission(
            messages,
            max_response_tokens=max_response_tokens,
            allow_functions=allow_functions,
        )

        logger.debug("------- RAW RESPONSE ----------")
        logger.debug(response["choices"])
        logger.debug("------- END RAW RESPONSE ----------")

        # If the response requests a function call, prefer this over the main response
        response_message = self.extract_response_message(response)
        if response_message is None:
            return RunResponse(
                raw_response=None,
                response=None,
                fix_transforms=None,
                function_call=None,
                function_arg=None,
            )

        function_call: Callable[[BaseModel], Any] | None = None
        function_parsed: BaseModel | None = None

        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_args_string = response_message["function_call"]["arguments"]
            if function_name not in self.functions:
                raise InvalidFunctionResponse(function_name)

            function_call = self.functions[function_name]
            function_arg_model = get_argument_for_function(function_call)

            # Parameters are formatted as raw json strings
            try:
                function_parsed = function_arg_model.model_validate_json(
                    function_args_string
                )
            except (ValueError, ValidationError):
                raise InvalidFunctionParameters(function_name, function_args_string)

        raw_response = GPTMessage.model_validate(response_message)
        raw_response.allow_templating = False

        extracted_json, fixed_payload = self.extract_json(
            response_message, self.extract_type
        )

        # Cast to schema model
        if extracted_json is None:
            return RunResponse(
                raw_response=raw_response,
                response=None,
                fix_transforms=fixed_payload,
                function_call=function_call,
                function_arg=function_parsed,
            )

        if not self.schema_model:
            raise ValueError(
                "GPTJSON failed to cast results into schema model. self.schema_model was unset."
            )

        # Allow pydantic to handle the validation
        return RunResponse(
            raw_response=raw_response,
            response=self.schema_model(**extracted_json),
            fix_transforms=fixed_payload,
            function_call=function_call,
            function_arg=function_parsed,
        )

    async def stream(
        self,
        messages: list[GPTMessage],
        max_response_tokens: int | None = None,
        format_variables: dict[str, Any] | None = None,
        truncation_options: TruncationOptions | None = None,
    ) -> AsyncIterator[StreamingObject[SchemaType]]:
        """
        See `run` for documentation. This method is an async generator wrapper around `run` that streams partial results
        instead of returning them all at once.

        :return: yields `StreamingObject[SchemaType]`s.
        """
        if self.extract_type != ResponseType.DICTIONARY:
            raise NotImplementedError(
                "For now, streaming is only supported for dictionary responses."
            )
        for field_type in self.schema_model.__annotations__.values():
            if field_type != str:
                raise NotImplementedError(
                    "For now, streaming is not supported for nested dictionary responses."
                )

        messages = self.fill_messages(
            messages, format_variables, truncation_options, max_response_tokens
        )

        # Most requests succeed on the first try but we wrap it locally here in case
        # there is some temporarily instability with the API. If there are longer periods
        # of instability, there should be system-wide retries in a daemon.
        backoff_request_submission = backoff.on_exception(
            backoff.expo,
            (RateLimitError, OpenAITimeout, APIConnectionError),
            max_tries=self.openai_max_retries,
            on_backoff=handle_backoff,
        )(self.submit_request)

        raw_responses = await backoff_request_submission(
            messages,
            max_response_tokens=max_response_tokens,
            stream=True,
            allow_functions=False,
        )

        previous_partial = None
        cumulative_response = ""
        async for raw_response in raw_responses:
            logger.debug(f"------- RAW RESPONSE ----------")
            logger.debug(raw_response)
            logger.debug(f"------- END RAW RESPONSE ----------")

            response = ChatCompletionChunk(**raw_response)

            if response.choices[0].delta.role is not None:
                # Ignore assistant role message
                continue
            if response.choices[0].finish_reason is not None:
                # Ignore finish message
                continue

            cumulative_response += response.choices[0].delta.content or ""

            # Unexpected but possible error if the schema has been unset
            if self.schema_model is None:
                raise ValueError(
                    "GPTJSON failed to cast results into schema model. self.schema_model was unset."
                )

            partial_data, proposed_event = parse_streamed_json(cumulative_response)
            partial_response = prepare_streaming_object(
                self.schema_model, partial_data, previous_partial, proposed_event
            )

            if (
                previous_partial is None
                or previous_partial.partial_obj != partial_response.partial_obj
            ):
                yield partial_response
                previous_partial = partial_response

    def extract_json(self, response_message, extract_type: ResponseType):
        """
        Assumes one main block of results, either list of dictionary
        """

        full_response = response_message["content"]
        if not full_response:
            return None, None

        extracted_response = find_json_response(full_response, extract_type)
        if extracted_response is None:
            return None, None

        # Save the original response before we start modifying it
        fixed_response = extracted_response
        fixed_response, fixed_truncation = fix_truncated_json(fixed_response)
        fixed_response, fixed_bools = fix_bools(fixed_response)

        fixed_payload = FixTransforms(
            fixed_bools=fixed_bools,
            fixed_truncation=fixed_truncation,
        )

        try:
            return json_loads(fixed_response), fixed_payload
        except JSONDecodeError as e:
            logger.debug(f"Extracted: {extracted_response}")
            logger.debug(f"Did parse: {fixed_response}")
            logger.error(f"JSON decode error, likely malformed json input: {e}")
            return None, fixed_payload

    def extract_response_message(self, completion_response):
        choices = completion_response["choices"]

        if not choices:
            logger.warning("No choices available, should report error...")
            return None

        return choices[0]["message"]

    async def submit_request(
        self,
        messages: list[GPTMessage],
        max_response_tokens: int | None,
        allow_functions: bool,
        stream: bool = False,
    ):
        """
        If a request times out, will raise an OpenAITimeout.

        """
        logger.debug("------- START MESSAGE ----------")
        logger.debug(messages)
        logger.debug("------- END MESSAGE ----------")
        if self.auto_trim:
            messages = self.trim_messages(messages, self.max_tokens)

        optional_parameters: dict[str, Any] = {}

        if max_response_tokens:
            optional_parameters["max_tokens"] = max_response_tokens

        if allow_functions and self.functions:
            optional_parameters["functions"] = [
                parse_function(fn) for fn in self.functions.values()
            ]
            optional_parameters["function_call"] = "auto"

        execute_prediction = openai.ChatCompletion.acreate(
            model=self.model,
            messages=[self.message_to_dict(message) for message in messages],
            temperature=self.temperature,
            api_key=self.api_key,
            stream=stream,
            **optional_parameters,
            **self.openai_arguments,
        )

        # The 'timeout' parameter supported by the OpenAI API is only used to cycle
        # the model while it's warming up
        # https://github.com/openai/openai-python/blob/fe3abd16b582ae784d8a73fd249bcdfebd5752c9/openai/api_resources/chat_completion.py#L41
        # We instead use a client-side timeout to prevent the API from hanging, which is more inline
        # with the expected behavior of our passed timeout.
        if self.timeout is None:
            return await execute_prediction
        else:
            try:
                return await wait_for(execute_prediction, timeout=self.timeout)
            except AsyncTimeoutError:
                raise OpenAITimeout

    def fill_messages(
        self,
        messages: list[GPTMessage],
        format_variables: dict[str, Any] | None,
        truncation_options: TruncationOptions | None,
        max_response_tokens: int | None,
    ):
        if truncation_options is None:
            return [
                self.fill_message_template(message, format_variables or {})
                for message in messages
            ]

        if (
            not format_variables
            or truncation_options.target_variable not in format_variables
        ):
            raise ValueError(
                f"Variable {truncation_options.target_variable} not found in message variables."
            )
        if truncation_options.max_prompt_tokens is None and max_response_tokens is None:
            raise ValueError(
                "Error in parsing truncation options: Either truncation_options.max_prompt_tokens or max_response_tokens must be set."
            )
        if (
            truncation_options.max_prompt_tokens is not None
            and truncation_options.max_prompt_tokens + (max_response_tokens or 0)
            > self.max_tokens
        ):
            raise ValueError(
                f"Truncation options max_prompt_tokens {truncation_options.max_prompt_tokens} plus max_response_tokens {max_response_tokens} exceeds model max tokens {self.max_tokens}."
            )

        # if max_prompt_tokens is not set, we synthetically determine the maximum
        # allowed amount of response tokens to keep the full prompt and response
        # within the model length bounds
        truncation_options.max_prompt_tokens = truncation_options.max_prompt_tokens or (
            self.max_tokens - (max_response_tokens or 0)
        )

        # fill the messages without the target variable to calculate the "space" we have left
        format_variables_no_target = format_variables.copy()
        format_variables_no_target[truncation_options.target_variable] = ""
        target_variable_max_tokens = (
            truncation_options.max_prompt_tokens
            - num_tokens_from_messages(
                [
                    self.message_to_dict(
                        self.fill_message_template(message, format_variables_no_target)
                    )
                    for message in messages
                ],
                self.model,
            )
        )

        if target_variable_max_tokens < 0:
            raise ValueError(
                f"Truncation options max_prompt_tokens {truncation_options.max_prompt_tokens} is too small to fit the messages."
            )

        truncated_target_variable = truncate_tokens(
            text=format_variables[truncation_options.target_variable],
            model=self.model,
            mode=truncation_options.truncation_mode,
            max_tokens=target_variable_max_tokens,
            custom_truncate_next=truncation_options.custom_truncate_next,
        )

        return [
            self.fill_message_template(
                message,
                {
                    **format_variables,
                    truncation_options.target_variable: truncated_target_variable,
                },
            )
            for message in messages
        ]

    def fill_message_template(
        self, message: GPTMessage, format_variables: dict[str, Any]
    ):
        auto_format = {
            SCHEMA_PROMPT_TEMPLATE_KEY: self.schema_prompt,
            SCHEMA_PROMPT_FUNCTION_KEY: json_dumps(
                [function_to_name(fn) for fn in self.functions.values()]
            ),
        }

        if message.content is None or not message.allow_templating:
            return message

        # Regular quotes should passthrough to the next stage, except for our special keys
        content = message.content.replace("{", "{{").replace("}", "}}")
        for key in auto_format.keys():
            content = content.replace("{{" + key + "}}", "{" + key + "}")
        content = content.format(**auto_format)

        # We do this formatting in a separate pass so we can fill any template variables that might
        # have been left in the pydantic field typehints
        content = content.format(**format_variables)

        new_message = copy(message)
        new_message.content = content
        return new_message

    def message_to_dict(self, message: GPTMessage):
        obj = json_loads(message.model_dump_json(exclude_unset=True))
        obj.pop("allow_templating", None)
        return obj

    def trim_messages(self, messages: list[GPTMessage], n: int):
        """
        Returns a list of messages with a total token count less than n tokens,
        cropping the last message if needed.

        Args:
            messages (list): List of strings to be checked.
            n (int): The maximum number of tokens allowed.

        Returns:
            list: A list of messages with a total token count less than n tokens.
        """
        message_text = [message.content for message in messages if message.content]

        enc = encoding_for_model("gpt-4")
        filtered_messages = []
        current_token_count = 0
        original_token_count = sum(
            [len(enc.encode(message)) for message in message_text]
        )

        for message in message_text:
            tokens = enc.encode(message)
            message_token_count = len(tokens)

            if current_token_count + message_token_count < n:
                filtered_messages.append(message)
                current_token_count += message_token_count
            else:
                remaining_tokens = n - current_token_count
                if remaining_tokens > 0:
                    cropped_message = enc.decode(tokens[:remaining_tokens])
                    filtered_messages.append(cropped_message)
                current_token_count += remaining_tokens
                break

        # Recreate the messages with our new text
        new_messages = []
        for i, content in enumerate(filtered_messages):
            new_message = copy(messages[i])
            new_message.content = content
            new_messages.append(new_message)

        # Log a copy of the message array if we have to crop it
        if current_token_count != original_token_count:
            logger.debug(
                f"Trimmed message from {original_token_count} to {current_token_count} tokens: {new_messages}",
            )
        else:
            logger.debug(
                f"Skipping trim ({original_token_count}) ({current_token_count})"
            )

        return new_messages

    def __class_getitem__(cls, item):
        cls._cls_schema_model = item
        return cls
