from dataclasses import dataclass
from enum import Enum, unique

from gpt_json.transformations import JsonFixEnum


@unique
class ResponseType(Enum):
    DICTIONARY = "DICTIONARY"
    LIST = "LIST"


@unique
class GPTMessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@unique
class GPTModelVersion(Enum):
    GPT_3_5 = "gpt-3.5-turbo"
    GPT_4 = "gpt-4-0314"


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
