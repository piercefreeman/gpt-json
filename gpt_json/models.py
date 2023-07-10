import sys
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable, Iterator

if sys.version_info >= (3, 11):
    from enum import StrEnum

    EnumSuper = StrEnum
else:
    EnumSuper = Enum


@unique
class ResponseType(EnumSuper):
    DICTIONARY = "DICTIONARY"
    LIST = "LIST"


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


@unique
class ModelProvider(EnumSuper):
    OPENAI = "oai"
    ANTHROPIC = "anthropic"

    @staticmethod
    def get_provider(model: str):
        if model.startswith("gpt-"):
            return ModelProvider.OPENAI
        elif model.startswith("claude-"):
            return ModelProvider.ANTHROPIC
        else:
            raise ValueError(f"Unknown model {model}")


@unique
class GPTModelVersion(EnumSuper):
    GPT_3_5 = "gpt-3.5-turbo"
    GPT_4 = "gpt-4-0314"
    CLAUDE = "claude-v1"
    CLAUDE_100K = "claude-v1-100k"


@unique
class JsonFixEnum(EnumSuper):
    UNCLOSED_OBJECT = "unclosed_object"
    UNCLOSED_KEY = "unclosed_key"
    UNCLOSED_VALUE = "unclosed_value"
    MISSING_VALUE = "missing_value"


@dataclass
class FixTransforms:
    """
    How a gpt payload was modified to be valid
    """

    fixed_truncation: JsonFixEnum | None = None
    fixed_bools: bool = False


@dataclass
class GPTMessage:
    """
    A single message in the chat sequence
    """

    role: GPTMessageRole
    content: str


@dataclass
class TruncationOptions:
    """
    Options for truncating the input variables
    """

    target_variable: str
    truncation_mode: VariableTruncationMode
    max_prompt_tokens: int | None = None
    custom_truncate_next: Callable[[str], str] | None = None
