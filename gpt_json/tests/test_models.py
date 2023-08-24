import sys
from enum import Enum

import pytest

import gpt_json.models as models_file
from gpt_json.models import GPTMessage, GPTMessageRole


@pytest.mark.parametrize("model_file", [models_file])
def test_string_enums(model_file):
    if sys.version_info < (3, 11):
        pytest.skip("Only Python 3.11+ has native support for string-based enums")
        return
    else:
        from enum import StrEnum

        found_enums = 0
        for obj in model_file.__dict__.values():
            if (
                isinstance(obj, type)
                and issubclass(obj, Enum)
                and not obj in {Enum, StrEnum}
            ):
                found_enums += 1
                assert issubclass(obj, StrEnum), f"{obj} is not a StrEnum"

        # Every file listed in pytest should have at least one enum
        assert found_enums > 0, f"No enums found in {model_file}"


def test_gpt_message_validates_function():
    with pytest.raises(ValueError):
        GPTMessage(
            role=GPTMessageRole.SYSTEM,
            name="function_name",
            content="function_content",
        )

    with pytest.raises(ValueError):
        GPTMessage(
            role=GPTMessageRole.FUNCTION,
            content="function_content",
        )

    GPTMessage(
        role=GPTMessageRole.FUNCTION,
        name="function_name",
        content="function_content",
    )
