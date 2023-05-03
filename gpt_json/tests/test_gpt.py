import pytest

from gpt_json.gpt import GPTJSON
from gpt_json.models import GPTMessage, GPTMessageRole


@pytest.mark.parametrize(
        "role_type,expected",
        [
            (GPTMessageRole.SYSTEM, "system"),
            (GPTMessageRole.USER, "user"),
            (GPTMessageRole.ASSISTANT, "assistant"),
        ]
)
def test_cast_message_to_gpt_format(role_type: GPTMessageRole, expected: str):
    parser = GPTJSON()
    assert parser.message_to_dict(
        GPTMessage(
            role=role_type,
            content="test",
        )
    )["role"] == expected
