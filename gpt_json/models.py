import sys
from base64 import b64encode
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum, unique
from typing import Callable, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

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


@dataclass
class ModelVersionParams:
    api_name: str
    max_length: int
    deprecated_date: date | None = None
    archived: bool = False


@unique
class GPTModelVersion(Enum):
    # Model versions prioritize explicit datestamped model versions over the generic
    # counterparts to reduce errors caused by invisible model-skew
    # https://platform.openai.com/docs/models/continuous-model-upgrades

    GPT_3_5_0613 = ModelVersionParams(
        api_name="gpt-3.5-turbo-0613",
        max_length=16_385,
        deprecated_date=date(2024, 6, 13),
    )
    GPT_3_5_1106 = ModelVersionParams(api_name="gpt-3.5-turbo-1106", max_length=16_385)
    GPT_3_5_0125 = ModelVersionParams(api_name="gpt-3.5-turbo-0125", max_length=16_385)

    GPT_4_0613 = ModelVersionParams(api_name="gpt-4-0613", max_length=8_192)
    GPT_4_32K_0613 = ModelVersionParams(api_name="gpt-4-32k-0613", max_length=32768)
    GPT_4_VISION_PREVIEW_1106 = ModelVersionParams(
        api_name="gpt-4-1106-vision-preview", max_length=128_000
    )

    GPT_4_O = ModelVersionParams(api_name="gpt-4o", max_length=128_000)
    GPT_4_O_MINI = ModelVersionParams(api_name="gpt-4o-mini", max_length=128_000)

    # Deprecated internally - switch to explicit model revisions
    # Kept for reverse compatibility
    GPT_3_5 = replace(GPT_3_5_0613, archived=True)  # type: ignore
    GPT_4 = replace(GPT_4_0613, archived=True)  # type: ignore


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


class ImageContent(BaseModel):
    class ImageUrl(BaseModel):
        url: HttpUrl

    class ImageBytes(BaseModel):
        url: str

        @field_validator("url", mode="after")
        def validate_url(cls, value):
            if isinstance(value, str):
                # Validate that we are a base64 encoded image
                if not value.startswith("data:"):
                    raise ValueError("Invalid image URL, must be a data URL")
            return value

    image_url: ImageUrl | ImageBytes

    payload_type: Literal["image_url"] = Field(default="image_url", alias="type")

    @classmethod
    def from_url(cls, url: str):
        return ImageContent(image_url=ImageContent.ImageUrl(url=HttpUrl(url)))

    @classmethod
    def from_bytes(cls, image_bytes: bytes, image_mime: str):
        encoded_image = b64encode(image_bytes).decode()
        data_url = f"data:{image_mime};base64,{encoded_image}"
        return ImageContent(image_url=ImageContent.ImageBytes(url=data_url))


class TextContent(BaseModel):
    text: str

    payload_type: Literal["text"] = Field(default="text", alias="type")


class GPTMessage(BaseModel):
    """
    A single message in the chat sequence
    """

    role: GPTMessageRole
    content: str | list[ImageContent | TextContent] | None

    @field_validator("content", mode="before")
    def wrap_content_in_payload(cls, value):
        # Support for old-syntax raw strings in the content
        if isinstance(value, str):
            return [TextContent(text=value)]
        return value

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

    def get_content_payloads(self) -> list[TextContent | ImageContent]:
        if isinstance(self.content, str):
            return [TextContent(text=self.content)]
        elif self.content is not None:
            return self.content
        else:
            return []


@dataclass
class TruncationOptions:
    """
    Options for truncating the input variables
    """

    target_variable: str
    truncation_mode: VariableTruncationMode
    max_prompt_tokens: int | None = None
    custom_truncate_next: Callable[[str], str] | None = None
