from dataclasses import replace
from json import loads as json_loads
from json.decoder import JSONDecodeError
from typing import Generic, List, Type, TypeVar, get_origin, get_args

import backoff
import openai
from openai.error import APIConnectionError, RateLimitError
from openai.error import Timeout as OpenAITimeout
from pydantic import BaseModel
from tiktoken import encoding_for_model
from typing import Any

from gpt_json.models import GPTMessage, GPTModelVersion, ResponseType
from gpt_json.parsers import find_json_response
from gpt_json.truncation import fix_truncated_json
from gpt_json.prompts import generate_schema_prompt

import logging

logger = logging.getLogger('my_logger')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    schema_model: Type[SchemaType] = None

    def __init__(
        self,
        api_key: str,
        model: GPTModelVersion | str = GPTModelVersion.GPT_4,
        auto_trim: bool = False,
        auto_trim_response_overhead: int = 0,
        # For messages that are relatively deterministic
        temperature = 0.2,
        timeout = 60,
    ):
        self.model = model.value if isinstance(model, GPTModelVersion) else model
        self.auto_trim = auto_trim
        self.temperature = temperature
        self.timeout = timeout

        if not self.schema_model:
            raise ValueError("GPTJSON needs to be instantiated with a schema model, like GPTJSON[MySchema](...args).")

        if get_origin(self.schema_model) in {list, List}:
            self.extract_type = ResponseType.LIST
        elif issubclass(self.schema_model, BaseModel):
            self.extract_type = ResponseType.DICTIONARY
        else:
            raise ValueError("GPTJSON needs to be instantiated with either a pydantic.BaseModel schema or a list of those schemas.")

        if self.auto_trim:
            if "gpt-4" in self.model:
                self.max_tokens = 8192 - auto_trim_response_overhead
            elif "gpt-3.5" in self.model:
                self.max_tokens = 4096 - auto_trim_response_overhead
            else:
                raise ValueError("Unknown model to infer max tokens, see https://platform.openai.com/docs/models/gpt-4 for more information on token length.")

        self.schema_prompt = generate_schema_prompt(self.schema_model)
        self.api_key = api_key

    async def run(
        self,
        messages: list[GPTMessage],
        max_tokens: int | None = None,
        format_variables: dict[str, Any] | None = None,
    ) -> SchemaType | None:
        messages = [
            self.fill_message_template(message, format_variables or {})
            for message in messages
        ]

        response = await self.submit_request(messages, max_tokens=max_tokens)
        logger.debug("------- RAW RESPONSE ----------")
        logger.debug(response["choices"])
        logger.debug("------- END RAW RESPONSE ----------")
        extracted_json = self.extract_json(response, self.extract_type)

        # Cast to schema model
        if extracted_json is None:
            return None

        # Allow pydantic to handle the validation
        if isinstance(extracted_json, list):
            model = get_args(self.schema_model)[0]
            return [model(**item) for item in extracted_json]
        else:
            return self.schema_model(**extracted_json)

    def extract_json(self, completion_response, extract_type: ResponseType):
        """
        Assumes one main block of results, either list of dictionary

        """
        choices = completion_response["choices"]

        if not choices:
            logger.warning("No choices available, should report error...")
            return None

        full_response = choices[0]["message"]["content"]

        extracted_response = find_json_response(full_response, extract_type)
        if extracted_response is None:
            return None

        extracted_response = extracted_response.replace("True", "true")
        extracted_response = extracted_response.replace("False", "false")

        fixed_response = fix_truncated_json(extracted_response)

        try:
            return json_loads(fixed_response)
        except JSONDecodeError as e:
            logger.debug("Extracted", extracted_response)
            logger.debug("Did parse", fixed_response)
            logger.error("JSON decode error, likely malformed json input", e)
            return None

    # Most requests succeed on the first try but we wrap it locally here in case
    # there is some temporarily instability with the API. If there are longer periods
    # of instability, there should be system-wide retries in a daemon.
    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, OpenAITimeout, APIConnectionError),
        max_tries=6,
        on_backoff=handle_backoff
    )
    async def submit_request(
        self,
        messages: list[GPTMessage],
        max_tokens: int | None,
    ):
        logger.debug("------- START MESSAGE ----------")
        logger.debug(messages)
        logger.debug("------- END MESSAGE ----------")
        if self.auto_trim:
            messages = self.trim_messages(messages, self.max_tokens)

        optional_parameters = {}

        if max_tokens:
            optional_parameters["max_tokens"] = max_tokens

        return await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                self.message_to_dict(message)
                for message in messages
            ],
            temperature=self.temperature,
            timeout=self.timeout,
            api_key=self.api_key,
            **optional_parameters,
        )

    def fill_message_template(self, message: GPTMessage, format_variables: dict[str, Any]):
        content = message.content.format(
            **{
                SCHEMA_PROMPT_TEMPLATE_KEY: self.schema_prompt,
            }
        )

        # We do this formatting in a separate pass so we can fill any template variables that might
        # have been left in the pydantic field typehints
        content = content.format(**format_variables)

        return replace(
            message,
            content=content,
        )

    def message_to_dict(self, message: GPTMessage):
        return {
            "role": message.role.value,
            "content": message.content
        }

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
        message_text = [
            message.content
            for message in messages
        ]

        enc = encoding_for_model("gpt-4")
        filtered_messages = []
        current_token_count = 0
        original_token_count = sum([
            len(enc.encode(message))
            for message in message_text
        ])

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
                f"Trimmed message from {original_token_count} to {current_token_count} tokens",
                new_messages,
            )
        else:
            logger.debug(f"Skipping trim ({original_token_count}) ({current_token_count})")

        return new_messages

    def __class_getitem__(cls, item):
        new_cls = super().__class_getitem__(item)
        new_cls.schema_model = item
        return new_cls
