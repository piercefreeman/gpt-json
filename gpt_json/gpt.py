import logging
from asyncio import TimeoutError as AsyncTimeoutError
from asyncio import wait_for
from dataclasses import replace
from json import loads as json_loads
from json.decoder import JSONDecodeError
from typing import (
    Any,
    AsyncIterator,
    Generic,
    List,
    Type,
    TypeVar,
    get_args,
    get_origin,
)

import anthropic
import backoff
import openai
from openai.error import APIConnectionError, RateLimitError
from openai.error import Timeout as OpenAITimeout
from pydantic import BaseModel
from tiktoken import encoding_for_model

from gpt_json.models import (
    FixTransforms,
    GPTMessage,
    GPTMessageRole,
    GPTModelVersion,
    ModelProvider,
    ResponseType,
    TruncationOptions,
    VariableTruncationMode,
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
from gpt_json.types_anthropic import AnthropicCompletion
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


SchemaType = TypeVar("SchemaType", bound=BaseModel)

SCHEMA_PROMPT_TEMPLATE_KEY = "json_schema"


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
        # For messages that are relatively deterministic
        temperature=0.2,
        api_max_retries=3,
        timeout: int | None = None,
        **kwargs,
    ):
        """
        :param api_key: OpenAI/Anthropic API key, if `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` environment variable is not set
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
        self.model_provider = ModelProvider.get_provider(self.model)
        self.auto_trim = auto_trim
        self.temperature = temperature
        self.timeout = timeout
        self.api_max_retries = api_max_retries
        self.api_arguments = kwargs
        self.schema_model = self._cls_schema_model
        self.__class__._cls_schema_model = None

        if "oai_max_retries" in kwargs:
            logging.warn(
                "oai_max_retries will be deprecated soon, please use api_max_retries instead."
            )
            self.api_max_retries = kwargs["oai_max_retries"]

        if not self.schema_model:
            raise ValueError(
                "GPTJSON needs to be instantiated with a schema model, like GPTJSON[MySchema](...args)."
            )

        if get_origin(self.schema_model) in {list, List}:
            self.extract_type = ResponseType.LIST
        elif issubclass(self.schema_model, BaseModel):
            self.extract_type = ResponseType.DICTIONARY
        else:
            raise ValueError(
                "GPTJSON needs to be instantiated with either a pydantic.BaseModel schema or a list of those schemas."
            )

        trim_tokens = auto_trim_response_overhead if auto_trim else 0
        if "gpt-4" in self.model:
            self.max_tokens = 8192 - trim_tokens
        elif "gpt-3.5" in self.model:
            self.max_tokens = 4096 - trim_tokens
        elif "claud-v1" in self.model:
            self.max_tokens = 9000 - trim_tokens
        elif "claud-v1-100k" in self.model:
            self.max_tokens = 100000 - trim_tokens
        else:
            raise ValueError(
                "Unknown model to infer max tokens, see https://platform.openai.com/docs/models/gpt-4 or https://console.anthropic.com/docs/api/reference for more information on token length."
            )

        self.schema_prompt = generate_schema_prompt(self.schema_model)
        self.api_key = api_key

    async def run(
        self,
        messages: list[GPTMessage],
        max_response_tokens: int | None = None,
        format_variables: dict[str, Any] | None = None,
        truncation_options: TruncationOptions | None = None,
    ) -> tuple[SchemaType | list[SchemaType], FixTransforms] | tuple[None, None]:
        """
        :param messages: List of GPTMessage objects to send to the API
        :param max_response_tokens: Maximum number of tokens allowed in the response
        :param format_variables: Variables to format into the message template. Uses standard
            Python string formatting, like "Hello {name}".format(name="World")

        :return: Tuple of (parsed response, fix transforms). The transformations here is a object that
            contains the allowed modifications that we might do to cleanup a GPT payload. It allows client callers
            to decide whether they want to allow the modifications or not.

        """
        if self.model_provider == ModelProvider.ANTHROPIC:
            if len(messages) != 1 or messages[0].role != GPTMessageRole.USER:
                raise NotImplementedError(
                    "For now, CLAUDE models only support one User message."
                )
            if truncation_options is not None:
                raise NotImplementedError(
                    "For now, CLAUDE models do not support truncation."
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
            max_tries=self.api_max_retries,
            on_backoff=handle_backoff,
        )(self.submit_request)

        response = await backoff_request_submission(
            messages, max_response_tokens=max_response_tokens
        )
        logger.debug("------- RAW RESPONSE ----------")
        logger.debug(
            response["choices"]
            if self.model_provider == ModelProvider.OPENAI
            else response["completion"]
        )
        logger.debug("------- END RAW RESPONSE ----------")
        extracted_json, fixed_payload = self.extract_json(response, self.extract_type)

        # Cast to schema model
        if extracted_json is None:
            return None, None

        if not self.schema_model:
            raise ValueError(
                "GPTJSON failed to cast results into schema model. self.schema_model was unset."
            )

        # Allow pydantic to handle the validation
        if isinstance(extracted_json, list):
            model = get_args(self.schema_model)[0]
            return [model(**item) for item in extracted_json], fixed_payload
        else:
            return self.schema_model(**extracted_json), fixed_payload

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
        if self.model_provider == ModelProvider.ANTHROPIC:
            raise NotImplementedError(
                "For now, streaming is only supported for the OpenAI API."
            )
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
            max_tries=self.api_max_retries,
            on_backoff=handle_backoff,
        )(self.submit_request)

        raw_responses = await backoff_request_submission(
            messages, max_response_tokens=max_response_tokens, stream=True
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

    def extract_json(self, completion_response, extract_type: ResponseType):
        """
        Assumes one main block of results, either list of dictionary
        """
        if self.model_provider == ModelProvider.OPENAI:
            choices = completion_response["choices"]

            if not choices:
                logger.warning("No choices available, should report error...")
                return None, None

            full_response = choices[0]["message"]["content"]
        elif self.model_provider == ModelProvider.ANTHROPIC:
            full_response = completion_response["completion"]

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

    async def submit_request(
        self,
        messages: list[GPTMessage],
        max_response_tokens: int | None,
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

        optional_parameters = {}

        if max_response_tokens and self.model_provider == ModelProvider.OPENAI:
            optional_parameters["max_tokens"] = max_response_tokens
        elif self.model_provider == ModelProvider.ANTHROPIC:
            # need to specify a default for CLAUDE models.
            # using OpenAI's default of 16 tokens: https://platform.openai.com/docs/api-reference/completions/create#completions/create-max_tokens
            optional_parameters["max_tokens_to_sample"] = (
                max_response_tokens if max_response_tokens else 16
            )

        if self.model_provider == ModelProvider.OPENAI:
            execute_prediction = openai.ChatCompletion.acreate(
                model=self.model,
                messages=[self.message_to_dict(message) for message in messages],
                temperature=self.temperature,
                api_key=self.api_key,
                stream=stream,
                **optional_parameters,
                **self.api_arguments,
            )
        elif self.model_provider == ModelProvider.ANTHROPIC:
            # format using Claude-mandated user prompt template
            # see more: https://console.anthropic.com/docs/api/reference#-v1-complete
            formatted_claude_prompt = (
                f"{anthropic.HUMAN_PROMPT} {messages[0].content}{anthropic.AI_PROMPT}"
            )
            execute_prediction = anthropic.Client(self.api_key).acompletion(
                prompt=formatted_claude_prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model=self.model,
                temperature=self.temperature,
                stream=stream,
                **optional_parameters,
                **self.api_arguments,
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
        }

        # Regular quotes should passthrough to the next stage, except for our special keys
        content = message.content.replace("{", "{{").replace("}", "}}")
        for key in auto_format.keys():
            content = content.replace("{{" + key + "}}", "{" + key + "}")
        content = content.format(**auto_format)

        # We do this formatting in a separate pass so we can fill any template variables that might
        # have been left in the pydantic field typehints
        content = content.format(**format_variables)

        return replace(
            message,
            content=content,
        )

    def message_to_dict(self, message: GPTMessage):
        return {"role": message.role.value, "content": message.content}

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
        message_text = [message.content for message in messages]

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
        new_messages = [
            replace(
                messages[i],
                content=content,
            )
            for i, content in enumerate(filtered_messages)
        ]

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
