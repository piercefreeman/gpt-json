import pytest

from gpt_json.gpt import GPTJSON
from gpt_json.models import GPTMessage, GPTMessageRole
from gpt_json.tests.shared import MySchema

def test_throws_error_if_no_model_specified():
    with pytest.raises(ValueError, match="needs to be instantiated with a schema model"):
        GPTJSON(None)


@pytest.mark.parametrize(
        "role_type,expected",
        [
            (GPTMessageRole.SYSTEM, "system"),
            (GPTMessageRole.USER, "user"),
            (GPTMessageRole.ASSISTANT, "assistant"),
        ]
)
def test_cast_message_to_gpt_format(role_type: GPTMessageRole, expected: str):
    parser = GPTJSON[MySchema](None)
    assert parser.message_to_dict(
        GPTMessage(
            role=role_type,
            content="test",
        )
    )["role"] == expected
