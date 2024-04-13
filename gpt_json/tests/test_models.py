import pytest

from gpt_json.models import GPTMessage, GPTMessageRole


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
