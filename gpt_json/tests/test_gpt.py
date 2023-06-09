import asyncio
from json import dumps as json_dumps
from time import time
from unittest.mock import AsyncMock, patch

import anthropic  # type: ignore
import openai
import pytest
from openai.error import Timeout as OpenAITimeout
from pydantic import BaseModel, Field

from gpt_json.gpt import GPTJSON
from gpt_json.models import FixTransforms, GPTMessage, GPTMessageRole, GPTModelVersion
from gpt_json.tests.shared import MySchema, MySubSchema
from gpt_json.transformations import JsonFixEnum


def test_throws_error_if_no_model_specified():
    with pytest.raises(
        ValueError, match="needs to be instantiated with a schema model"
    ):
        GPTJSON(None)


@pytest.mark.parametrize(
    "role_type,expected",
    [
        (GPTMessageRole.SYSTEM, "system"),
        (GPTMessageRole.USER, "user"),
        (GPTMessageRole.ASSISTANT, "assistant"),
    ],
)
def test_cast_message_to_gpt_format(role_type: GPTMessageRole, expected: str):
    parser = GPTJSON[MySchema](None)
    assert (
        parser.message_to_dict(
            GPTMessage(
                role=role_type,
                content="test",
            )
        )["role"]
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schema_typehint,response_raw,parsed,expected_transformations",
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
                sub_element=MySubSchema(name="Test"),
                reason=True,
            ),
            FixTransforms(),
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
                    sub_element=MySubSchema(name="Test"),
                    reason=True,
                )
            ],
            FixTransforms(),
        ),
        (
            MySchema,
            """
            Your response is as follows:
            {
                "text": "Test",
                "numerical": 123,
                "reason": True,
                "sub_element": {
                    "name": "Test"
                },
                "items": ["Item 1", "Item 2

            """,
            MySchema(
                text="Test",
                items=["Item 1", "Item 2"],
                numerical=123,
                sub_element=MySubSchema(name="Test"),
                reason=True,
            ),
            FixTransforms(
                fixed_bools=True, fixed_truncation=JsonFixEnum.UNCLOSED_VALUE
            ),
        ),
    ],
)
async def test_acreate(
    schema_typehint, response_raw, parsed, expected_transformations: FixTransforms
):
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
    with patch.object(openai.ChatCompletion, "acreate") as mock_acreate:
        # Make the mock function asynchronous
        mock_acreate.return_value = mock_response

        # Call the function and pass the expected parameters
        response, transformations = await model.run(messages=messages)

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
            api_key=None,
            stream=False,
        )

    assert response == parsed
    assert transformations == expected_transformations


@pytest.mark.asyncio
async def test_anthropic_model():
    model_version = GPTModelVersion.CLAUDE
    messages = [
        GPTMessage(
            role=GPTMessageRole.USER,
            content="Input prompt",
        )
    ]

    class TestSchema(BaseModel):
        text: str

    model = GPTJSON[TestSchema](
        None,
        model=model_version,
        temperature=0.0,
        timeout=60,
    )

    # Define mock response
    mock_response = {"completion": """{"text": "Test"} """, "stop_reason": "max_tokens"}

    # Create the mock
    with patch.object(anthropic.Client, "acompletion") as mock_acreate:
        # Make the mock function asynchronous
        mock_acreate.return_value = mock_response

        # Call the function and pass the expected parameters
        response, transformations = await model.run(
            messages=messages, max_response_tokens=10
        )

        # Assert that the mock function was called with the expected parameters
        mock_acreate.assert_called_with(
            prompt=f"{anthropic.HUMAN_PROMPT} {messages[0].content}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=model_version.value,
            temperature=0.0,
            stream=False,
            max_tokens_to_sample=10,
        )

    assert response == TestSchema(text="Test")
    assert transformations == FixTransforms()


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
                GPTMessage(role=GPTMessageRole.USER, content="World" * (8192 - 1)),
            ],
        ),
    ],
)
def test_trim_messages(input_messages, expected_output_messages):
    gpt = GPTJSON[MySchema](None, auto_trim=True, auto_trim_response_overhead=0)

    output_messages = gpt.trim_messages(input_messages, n=8192)

    assert len(output_messages) == len(expected_output_messages)

    for output_message, expected_output_message in zip(
        output_messages, expected_output_messages
    ):
        assert output_message.role == expected_output_message.role
        assert output_message.content == expected_output_message.content


def test_two_gptjsons():
    class TestSchema1(BaseModel):
        field1: str

    class TestSchema2(BaseModel):
        field2: str

    gptjson1 = GPTJSON[TestSchema1](None)

    # Shouldn't allow instantion without a schema
    # We already expect a mypy error here, which is why we need a `type ignore`
    # butr we also want to make sure that the error is raised at runtime
    with pytest.raises(ValueError):
        gptjson2 = GPTJSON(None)  # type: ignore

    gptjson2 = GPTJSON[TestSchema2](None)

    assert gptjson1.schema_model == TestSchema1
    assert gptjson2.schema_model == TestSchema2


def test_fill_message_template():
    class TestTemplateSchema(BaseModel):
        template_field: str = Field(description="Max length {max_length}")

    gpt = GPTJSON[TestTemplateSchema](None)
    assert gpt.fill_message_template(
        GPTMessage(
            role=GPTMessageRole.USER,
            content="Variable: {max_length}\nMy schema is here: {json_schema}",
        ),
        dict(
            max_length=100,
        ),
    ) == GPTMessage(
        role=GPTMessageRole.USER,
        content='Variable: 100\nMy schema is here: {\n"template_field": str // Max length 100\n}',
    )


@pytest.mark.asyncio
async def test_extracted_json_is_None():
    gpt = GPTJSON[MySchema](None)

    with patch.object(
        gpt,
        "submit_request",
        return_value={"choices": [{"message": {"content": "some content"}}]},
    ), patch.object(
        gpt, "extract_json", return_value=(None, FixTransforms(None, False))
    ):
        result, _ = await gpt.run(
            [GPTMessage(GPTMessageRole.SYSTEM, "message content")]
        )
        assert result is None


@pytest.mark.asyncio
async def test_no_valid_results_from_remote_request():
    gpt = GPTJSON[MySchema](None)

    with patch.object(gpt, "submit_request", return_value={"choices": []}):
        result, _ = await gpt.run(
            [GPTMessage(GPTMessageRole.SYSTEM, "message content")]
        )
        assert result is None


@pytest.mark.asyncio
async def test_unable_to_find_valid_json_payload():
    gpt = GPTJSON[MySchema](None)

    with patch.object(
        gpt,
        "submit_request",
        return_value={"choices": [{"message": {"content": "some content"}}]},
    ), patch.object(
        gpt, "extract_json", return_value=(None, FixTransforms(None, False))
    ):
        result, _ = await gpt.run(
            [GPTMessage(GPTMessageRole.SYSTEM, "message content")]
        )
        assert result is None


@pytest.mark.asyncio
async def test_unknown_model_to_infer_max_tokens():
    with pytest.raises(ValueError):
        GPTJSON[MySchema](model="UnknownModel", auto_trim=True)


@pytest.mark.asyncio
async def test_timeout():
    class MockResponse:
        """
        We need to build an actual response class here because the internal openai
        code postprocesses with the aiohttp response.

        """

        def __init__(self, response_text: str):
            self.status = 200
            self.headers: dict[str, str] = {}
            self.response_text = response_text

        async def read(self):
            mock_response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": self.response_text,
                        },
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ]
            }
            return json_dumps(mock_response).encode()

    with patch("aiohttp.ClientSession.request", new_callable=AsyncMock) as mock_request:
        # Mock a stalling request
        async def side_effect(*args, **kwargs):
            await asyncio.sleep(4)
            return MockResponse("TEST_RESPONSE")

        mock_request.side_effect = side_effect

        gpt = GPTJSON[MySchema](api_key="ABC", timeout=2)

        start_time = time()

        with pytest.raises(OpenAITimeout):
            await gpt.run(
                [GPTMessage(GPTMessageRole.SYSTEM, "message content")],
            )

        end_time = time()
        duration = end_time - start_time

        # Assert duration is about 2 seconds
        pytest.approx(duration, 2, 0.2)
