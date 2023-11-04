import sys
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable

from pydantic import BaseModel, model_validator

if sys.version_info >= (3, 11):
    from enum import StrEnum

    EnumSuper = StrEnum
else:
    EnumSuper = Enum


@unique
class ResponseType(EnumSuper):
    DICTIONARY = "DICTIONARY"


@unique
class VariableTruncationMode(EnumSuper):
    BEGINNING = "BEGINNING"
    TRAILING = "TRAILING"
    MIDDLE = "MIDDLE"
    RANDOM = "RANDOM"
    CUSTOM = "CUSTOM"


@unique
class GPTMessageRole(EnumSuper):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@unique
class GPTModelVersion(EnumSuper):
    GPT_3_5 = "gpt-3.5-turbo-0613"
    GPT_4 = "gpt-4-0613"


@unique
class JsonFixEnum(EnumSuper):
    UNCLOSED_OBJECT = "unclosed_object"
    UNCLOSED_KEY = "unclosed_key"
    UNCLOSED_VALUE = "unclosed_value"
    MISSING_VALUE = "missing_value"

    # Drop any additional JSON tags that occur after the main payload
    # has been processed; this most often happens when the models spit back
    # double close brackets like ]] or }}
    DROP_TRAILING_JSON = "drop_trailing_json"


@dataclass
class FixTransforms:
    """
    How a gpt payload was modified to be valid
    """

    fixed_truncation: JsonFixEnum | None = None
    fixed_bools: bool = False


class FunctionCall(BaseModel):
    arguments: str
    name: str


class GPTMessage(BaseModel):
    """
    A single message in the chat sequence
    """

    role: GPTMessageRole
    content: str | None

    # Name is only supported if we're formatting a function message
    name: str | None = None

    # Message from the server
    function_call: FunctionCall | None = None

    # If enabled, gpt-json will attempt to format the message with the runtime variables
    # Disable this in cases where you want the message to be formatted 1:1 with the input
    allow_templating: bool = True

    @model_validator(mode="after")
    def check_name_if_function(self):
        if self.role == GPTMessageRole.FUNCTION and self.name is None:
            raise ValueError("Must provide a name for function messages")
        if self.role != GPTMessageRole.FUNCTION and self.name is not None:
            raise ValueError("Cannot provide a name for non-function messages")
        return self


@dataclass
class TruncationOptions:
    """
    Options for truncating the input variables
    """

    target_variable: str
    truncation_mode: VariableTruncationMode
    max_prompt_tokens: int | None = None
    custom_truncate_next: Callable[[str], str] | None = None
