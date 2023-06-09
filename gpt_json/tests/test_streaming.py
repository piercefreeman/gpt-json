import json
from unittest.mock import patch

import openai
import pytest

from gpt_json.gpt import GPTJSON
from gpt_json.models import GPTMessage, GPTMessageRole, GPTModelVersion
from gpt_json.streaming import StreamingObject
from gpt_json.tests.utils.streaming_utils import tokenize
from gpt_json.tests.utils.test_streaming_utils import (
    EXAMPLE_DICT,
    EXAMPLE_DICT_STREAM_DATA,
    EXAMPLE_MULTI_NESTED,
    EXAMPLE_MULTI_NESTED_STREAM_DATA,
    EXAMPLE_STR_DICT,
    EXAMPLE_STR_DICT_STREAM_DATA,
    EXAMPLE_STR_LIST,
    EXAMPLE_STR_LIST_STREAM_DATA,
    ExampleDictSchema,
    ExampleMultiNestedSchema,
    ExampleStrDictSchema,
    ExampleStrListSchema,
)

MOCK_ASSISTANT_CHUNK = {
    "id": "chatcmpl-7GWTw9HlmVFOiXyWNBfNKVFzA55yy",
    "object": "chat.completion.chunk",
    "created": 1684172464,
    "model": "gpt-4-0314",
    "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}],
}
MOCK_FINISH_REASON_CHUNK = {
    "id": "chatcmpl-7GWTw9HlmVFOiXyWNBfNKVFzA55yy",
    "object": "chat.completion.chunk",
    "created": 1684172464,
    "model": "gpt-4-0314",
    "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
}
MOCK_CONTENT_CHUNK = lambda content: {
    "id": "chatcmpl-7GWTw9HlmVFOiXyWNBfNKVFzA55yy",
    "object": "chat.completion.chunk",
    "created": 1684172464,
    "model": "gpt-4-0314",
    "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}],
}


def _mock_oai_streaming_chunks(
    full_object, json_indent_level=2, prefix_str=None, postfix_str=None
):
    yield MOCK_ASSISTANT_CHUNK

    full_content = f"{prefix_str if prefix_str else ''}{json.dumps(full_object, indent=json_indent_level)}{postfix_str if postfix_str else ''}"
    full_content_tokens = tokenize(full_content)

    for token in full_content_tokens:
        yield MOCK_CONTENT_CHUNK(token)

    yield MOCK_FINISH_REASON_CHUNK


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "full_object,schema_typehint,expected_stream_data,should_support",
    [
        (EXAMPLE_STR_DICT, ExampleStrDictSchema, EXAMPLE_STR_DICT_STREAM_DATA, True),
        # TODO: support these cases in v1
        (EXAMPLE_DICT, ExampleDictSchema, EXAMPLE_DICT_STREAM_DATA, False),
        (EXAMPLE_STR_LIST, ExampleStrListSchema, EXAMPLE_STR_LIST_STREAM_DATA, False),
        (
            EXAMPLE_MULTI_NESTED,
            ExampleMultiNestedSchema,
            EXAMPLE_MULTI_NESTED_STREAM_DATA,
            False,
        ),
    ],
)
async def test_gpt_stream(
    full_object,
    schema_typehint,
    expected_stream_data,
    should_support,
):
    model_version = GPTModelVersion.GPT_3_5
    messages = [
        GPTMessage(
            role=GPTMessageRole.USER,
            content="Input prompt",
        )
    ]

    model = GPTJSON[schema_typehint](  # type: ignore
        None,
        model=model_version,
        temperature=0.0,
        timeout=60,
    )

    # Define mock response
    mocked_oai_raw_responses = _mock_oai_streaming_chunks(full_object)

    async def async_list_to_generator(my_list):
        for item in my_list:
            yield item

    mock_response = async_list_to_generator(mocked_oai_raw_responses)

    # Create the mock
    with patch.object(openai.ChatCompletion, "acreate") as mock_acreate:
        # Make the mock function asynchronous
        mock_acreate.return_value = mock_response

        if not should_support:
            with pytest.raises(NotImplementedError):
                streaming_objects = [
                    obj async for obj in model.stream(messages=messages)
                ]
            return True

        idx = 0
        async for stream_obj in model.stream(messages=messages):
            (
                expected_partial_obj,
                expected_event,
                expected_update_key,
                expected_value_change,
            ) = expected_stream_data[idx]
            expected_obj = StreamingObject[schema_typehint](  # type: ignore
                partial_obj=schema_typehint(**expected_partial_obj),
                event=expected_event,
                updated_key=expected_update_key,
                value_change=expected_value_change,
            )
            assert stream_obj == expected_obj

            idx += 1

        # Assert that the mock function was called with the expected parameters, including streaming
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
            stream=True,
            api_key=None,
        )
