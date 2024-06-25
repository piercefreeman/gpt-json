import asyncio
from json import dumps as json_dumps
from time import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from openai._exceptions import APITimeoutError as OpenAITimeout
from pydantic import BaseModel, Field
from pytest_httpx import HTTPXMock

from gpt_json.gpt import GPTJSON, ListResponse
from gpt_json.models import (
    FixTransforms,
    GPTMessage,
    GPTMessageRole,
    GPTModelVersion,
    TextContent,
)
from gpt_json.tests.shared import (
    GetCurrentWeatherRequest,
    MySchema,
    MySubSchema,
    UnitType,
    get_current_weather,
)
from gpt_json.transformations import JsonFixEnum


def make_assistant_response(choices: list[Any]):
    # https://platform.openai.com/docs/api-reference/chat/create
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0125",
        "system_fingerprint": "fp_3478aj6f3a",
        "choices": choices,
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


def test_throws_error_if_no_model_specified():
    with pytest.raises(
        ValueError, match="needs to be instantiated with a schema model"
    ):
        GPTJSON(api_key="TEST")


@pytest.mark.parametrize(
    "role_type,expected",
    [
        (GPTMessageRole.SYSTEM, "system"),
        (GPTMessageRole.USER, "user"),
        (GPTMessageRole.ASSISTANT, "assistant"),
    ],
)
def test_cast_message_to_gpt_format(role_type: GPTMessageRole, expected: str):
    parser = GPTJSON[MySchema](api_key="TEST")
    assert (
        parser.message_to_dict(
            GPTMessage(
                role=role_type,
                content=[TextContent(text="test")],
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
            ListResponse[MySchema],
            """
            Your response is as follows:
            {
                "items": [
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
            }
            Your response is above.
            """,
            # Slight hack to work around ListResponse being a generic base that Pydantic can't
            # otherwise validate / output to a dictionary
            ListResponse[MySchema](
                items=[
                    MySchema(
                        text="Test",
                        items=["Item 1", "Item 2"],
                        numerical=123,
                        sub_element=MySubSchema(name="Test"),
                        reason=True,
                    )
                ]
            ),
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
async def test_create(
    schema_typehint,
    httpx_mock: HTTPXMock,
    response_raw: str,
    parsed: BaseModel,
    expected_transformations: FixTransforms,
):
    model_version = GPTModelVersion.GPT_4
    messages = [
        GPTMessage(
            role=GPTMessageRole.USER,
            content=[TextContent(text="Input prompt")],
        )
    ]

    # Define mock response
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json=make_assistant_response(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_raw,
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ]
        ),
    )

    model = GPTJSON[schema_typehint](
        api_key="TEST",
        model=model_version,
        temperature=0.0,
        timeout=60,
    )

    # Call the function and pass the expected parameters
    response = await model.run(messages=messages)

    assert response
    assert response.response
    assert response.response.model_dump() == parsed.model_dump()
    assert response.fix_transforms == expected_transformations


@pytest.mark.asyncio
async def test_create_with_function_calls(
    httpx_mock: HTTPXMock,
):
    model_version = GPTModelVersion.GPT_4
    messages = [
        GPTMessage(
            role=GPTMessageRole.USER,
            content=[TextContent(text="Input prompt")],
        )
    ]

    # Define mock response
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json=make_assistant_response(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": "get_current_weather",
                            "arguments": json_dumps(
                                {
                                    "location": "Boston",
                                    "unit": "fahrenheit",
                                }
                            ),
                        },
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ]
        ),
    )

    model = GPTJSON[MySchema](
        api_key="TEST",
        model=model_version,
        temperature=0.0,
        timeout=60,
        functions=[get_current_weather],
    )

    response = await model.run(messages=messages)

    assert response
    assert response.response is None
    assert response.function_call == get_current_weather
    assert response.function_arg == GetCurrentWeatherRequest(
        location="Boston", unit=UnitType.FAHRENHEIT
    )


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
    gpt = GPTJSON[MySchema](
        api_key="TEST", auto_trim=True, auto_trim_response_overhead=0
    )

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

    gptjson1 = GPTJSON[TestSchema1](api_key="TRUE")

    # Shouldn't allow instantion without a schema
    # We already expect a mypy error here, which is why we need a `type ignore`
    # butr we also want to make sure that the error is raised at runtime
    with pytest.raises(ValueError):
        gptjson2 = GPTJSON(None)  # type: ignore

    gptjson2 = GPTJSON[TestSchema2](api_key="TRUE")

    assert gptjson1.schema_model == TestSchema1
    assert gptjson2.schema_model == TestSchema2


def test_fill_message_schema_template():
    class TestTemplateSchema(BaseModel):
        template_field: str = Field(description="Max length {max_length}")

    gpt = GPTJSON[TestTemplateSchema](api_key="TRUE")
    assert gpt.fill_message_template(
        GPTMessage(
            role=GPTMessageRole.USER,
            content=[
                TextContent(
                    text="Variable: {max_length}\nMy schema is here: {json_schema}"
                )
            ],
        ),
        dict(
            max_length=100,
        ),
    ) == GPTMessage(
        role=GPTMessageRole.USER,
        content=[
            TextContent(
                text='Variable: 100\nMy schema is here: {\n"template_field": str // Max length 100\n}'
            )
        ],
    )


def test_fill_message_functions_template():
    class TestTemplateSchema(BaseModel):
        template_field: str = Field(description="Max length {max_length}")

    gpt = GPTJSON[TestTemplateSchema](api_key="TRUE", functions=[get_current_weather])
    assert gpt.fill_message_template(
        GPTMessage(
            role=GPTMessageRole.USER,
            content=[TextContent(text="Here are the functions available: {functions}")],
        ),
        format_variables=dict(),
    ) == GPTMessage(
        role=GPTMessageRole.USER,
        content=[
            TextContent(
                text='Here are the functions available: ["get_current_weather"]'
            )
        ],
    )


@pytest.mark.asyncio
async def test_extracted_json_is_none(httpx_mock: HTTPXMock):
    gpt = GPTJSON[MySchema](api_key="TRUE")

    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json=make_assistant_response(
            [{"message": {"content": "some content", "role": "assistant"}}]
        ),
    )

    with patch.object(
        gpt, "extract_json", return_value=(None, FixTransforms(None, False))
    ):
        result = await gpt.run(
            [
                GPTMessage(
                    role=GPTMessageRole.SYSTEM,
                    content=[TextContent(text="message content")],
                )
            ]
        )
        assert result.response is None


@pytest.mark.asyncio
async def test_no_valid_results_from_remote_request(
    httpx_mock: HTTPXMock,
):
    gpt = GPTJSON[MySchema](api_key="TRUE")

    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json=make_assistant_response([]),
    )

    result = await gpt.run(
        [
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=[TextContent(text="message content")],
            )
        ]
    )
    assert result.response is None


@pytest.mark.asyncio
async def test_unable_to_find_valid_json_payload(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json=make_assistant_response(
            [{"message": {"content": "some content", "role": "assistant"}}]
        ),
    )
    gpt = GPTJSON[MySchema](api_key="TRUE")

    with patch.object(
        gpt, "extract_json", return_value=(None, FixTransforms(None, False))
    ):
        result = await gpt.run(
            [
                GPTMessage(
                    role=GPTMessageRole.SYSTEM,
                    content=[TextContent(text="message content")],
                )
            ]
        )
        assert result.response is None


@pytest.mark.asyncio
async def test_unknown_model_to_infer_max_tokens():
    with pytest.raises(ValueError):
        GPTJSON[MySchema](api_key="TRUE", model="UnknownModel", auto_trim=True)


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

    with patch("gpt_json.gpt.AsyncOpenAI") as mock_client:
        # Mock a stalling request
        async def side_effect(*args, **kwargs):
            await asyncio.sleep(4)
            return MockResponse("TEST_RESPONSE")

        mock_client.return_value.chat.completions.create = AsyncMock(
            side_effect=side_effect
        )

        gpt = GPTJSON[MySchema](api_key="ABC", timeout=2)

        start_time = time()

        with pytest.raises(OpenAITimeout):
            await gpt.run(
                [
                    GPTMessage(
                        role=GPTMessageRole.SYSTEM,
                        content=[TextContent(text="message content")],
                    )
                ],
            )

        end_time = time()
        duration = end_time - start_time

        # Assert duration is about 2 seconds
        pytest.approx(duration, 2, 0.2)
