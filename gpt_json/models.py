import sys
from dataclasses import dataclass
from enum import Enum, unique

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
class GPTMessageRole(EnumSuper):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@unique
class GPTModelVersion(EnumSuper):
    GPT_3_5 = "gpt-3.5-turbo"
    GPT_4 = "gpt-4-0314"


@dataclass
class FixTransforms:
    """
    How a gpt payload was modified to be valid
    """

    fixed_truncation: bool = False
    fixed_bools: bool = False


@dataclass
class GPTMessage:
    """
    A single message in the chat sequence
    """

    role: GPTMessageRole
    content: str
