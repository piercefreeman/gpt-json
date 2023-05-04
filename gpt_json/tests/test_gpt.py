from unittest.mock import MagicMock, patch

import openai
import pytest

from gpt_json.gpt import GPTJSON
from gpt_json.models import GPTMessage, GPTMessageRole, GPTModelVersion
from gpt_json.tests.shared import MySchema, MySubSchema


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schema_typehint,response_raw,parsed",
    [
        (
            MySchema,
            """
            Your response is as follows:
            {
                "text": "Test",
                "items": ["Item 1", "Item 2"],
                "numerical": 123,
                "sub_element": {
                    "name": "Test"
                },
                "reason": true
            }
            Your response is above.
            """,
            MySchema(
                text="Test",
                items=["Item 1", "Item 2"],
                numerical=123,
                sub_element=MySubSchema(
                    name="Test"
                ),
                reason=True,
            )
        ),
        (
            list[MySchema],
            """
            Your response is as follows:
            [
                {
                    "text": "Test",
                    "items": ["Item 1", "Item 2"],
                    "numerical": 123,
                    "sub_element": {
                        "name": "Test"
                    },
                    "reason": true
                }
            ]
            Your response is above.
            """,
            [
                MySchema(
                    text="Test",
                    items=["Item 1", "Item 2"],
                    numerical=123,
                    sub_element=MySubSchema(
                        name="Test"
                    ),
                    reason=True,
                )                
            ]
        )
    ]
)
async def test_acreate(schema_typehint, response_raw, parsed):
    model_version = GPTModelVersion.GPT_3_5
    messages = [
        GPTMessage(
            role=GPTMessageRole.USER,
            content="Input prompt",
        )
    ]

    model = GPTJSON[schema_typehint](
        None,
        model=model_version,
        temperature=0.0,
        timeout=60,
    )

    # Define mock response
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response_raw,
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ]
    }

    # Create the mock
    with patch.object(openai.ChatCompletion, "acreate", return_value=mock_response) as mock_acreate:
        # Make the mock function asynchronous
        mock_acreate.__aenter__.return_value = MagicMock()
        mock_acreate.__aexit__.return_value = MagicMock()
        mock_acreate.__aenter__.return_value.__aenter__ = MagicMock(return_value=mock_response)

        # Call the function and pass the expected parameters
        response = await model.run(messages=messages)

        # Assert that the mock function was called with the expected parameters
        mock_acreate.assert_called_with(
            model=model_version.value,
            messages=[
                {
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in messages
            ],
            temperature=0.0,
            timeout=60,
            api_key=None,
        )

    assert response == parsed


@pytest.mark.parametrize(
    "input_messages,expected_output_messages",
    [
        # Messages fit within max_tokens, no change expected
        (
            [
                GPTMessage(role=GPTMessageRole.SYSTEM, content="Hello"),
                GPTMessage(role=GPTMessageRole.USER, content="World!"),
            ],
            [
                GPTMessage(role=GPTMessageRole.SYSTEM, content="Hello"),
                GPTMessage(role=GPTMessageRole.USER, content="World!"),
            ],
        ),
        # All messages trimmed to fit max_tokens
        (
            [
                GPTMessage(role=GPTMessageRole.SYSTEM, content="Hello"),
                GPTMessage(role=GPTMessageRole.USER, content="World" * 10000),
            ],
            [
                GPTMessage(role=GPTMessageRole.SYSTEM, content="Hello"),
                GPTMessage(role=GPTMessageRole.USER, content="World" * (8192-1)),
            ],
        ),
    ],
)
def test_trim_messages(input_messages, expected_output_messages):
    gpt = GPTJSON[MySchema](None, auto_trim=True, auto_trim_response_overhead=0)

    output_messages = gpt.trim_messages(input_messages, n=8192)

    assert len(output_messages) == len(expected_output_messages)

    for output_message, expected_output_message in zip(output_messages, expected_output_messages):
        assert output_message.role == expected_output_message.role
        assert output_message.content == expected_output_message.content
