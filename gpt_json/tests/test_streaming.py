import json
from typing import Any, List
from unittest.mock import MagicMock, patch

import openai
import pytest
import tiktoken
from pydantic import BaseModel

from gpt_json.gpt import GPTJSON
from gpt_json.models import GPTMessage, GPTMessageRole, GPTModelVersion
from gpt_json.streaming import StreamEventEnum, StreamingObject

enc = tiktoken.get_encoding("cl100k_base")

def _tokenize(text):
    return [enc.decode([tok]) for tok in enc.encode(text)]

def _tuple_merge(t1, t2):
    t1 = (t1,) if not isinstance(t1, tuple) else t1
    t2 = (t2,) if not isinstance(t2, tuple) else t2

    return t1 + t2

def _get_expected_stream_partial_objs(full_object):
    """This function implements the semantics of the behavior we expect from GPTJSON.stream().
    It is only used for testing and demonstrative purposes. 
    """
    if isinstance(full_object, list):
        value_iterators = [_get_expected_stream_partial_objs(v) for v in full_object]
        outer_partial = []
        yield outer_partial.copy(), StreamEventEnum.OBJECT_CREATED, None, None
        for key in range(len(full_object)):
            for inner_partial, event, inner_key, value_change in value_iterators[key]:
                if event == StreamEventEnum.OBJECT_CREATED:
                    outer_partial.append(inner_partial)
                    continue
                outer_partial[key] = inner_partial
                yield_key = _tuple_merge(len(outer_partial) - 1, inner_key)
                yield_key = len(outer_partial) - 1 
                if inner_key is not None:
                    yield_key = _tuple_merge(yield_key, inner_key)
                yield outer_partial.copy(), event, yield_key, value_change
    elif isinstance(full_object, dict):
        value_iterators = {
            k: _get_expected_stream_partial_objs(v) for k, v in full_object.items()
        }
        # get initial OBJECT_CREATED values for all keys first 
        outer_partial = {
            k : next(v)[0] for k, v in value_iterators.items()
        }
        yield outer_partial.copy(), StreamEventEnum.OBJECT_CREATED, None, None
        for key in full_object.keys():
            for inner_partial, event, inner_key, value_change in value_iterators[key]:
                # note: we've already consumed the first value (OBJECT_CREATED) for each key
                outer_partial[key] = inner_partial
                yield_key = key
                if inner_key is not None:
                    yield_key = _tuple_merge(yield_key, inner_key)
                yield outer_partial.copy(), event, yield_key, value_change
    elif isinstance(full_object, str):
        outer_partial = ""
        yield outer_partial, StreamEventEnum.OBJECT_CREATED, None, None

        tokens = _tokenize(full_object)
        for idx, token in enumerate(tokens):
            outer_partial += token
            event = StreamEventEnum.KEY_COMPLETED if idx == len(tokens) - 1 else StreamEventEnum.KEY_UPDATED
            yield outer_partial, event, None, token
    else: # number, bool
        yield None, StreamEventEnum.OBJECT_CREATED, None, None
        yield full_object, StreamEventEnum.KEY_COMPLETED, None, full_object

class ExampleStrDictSchema(BaseModel):
    student_model: str
    tutor_response: str

EXAMPLE_STR_DICT = {'student_model': 'The student.', 'tutor_response': 'Good!' }

EXAMPLE_STR_DICT_STREAM_DATA = [
    ({'student_model': '', 'tutor_response': ''}, StreamEventEnum.OBJECT_CREATED, None, None),
    ({'student_model': 'The', 'tutor_response': ''}, StreamEventEnum.KEY_UPDATED, 'student_model', 'The'),
    ({'student_model': 'The student', 'tutor_response': ''}, StreamEventEnum.KEY_UPDATED, 'student_model', ' student'),
    ({'student_model': 'The student.', 'tutor_response': ''}, StreamEventEnum.KEY_COMPLETED, 'student_model', '.'),
    ({'student_model': 'The student.', 'tutor_response': 'Good'}, StreamEventEnum.KEY_UPDATED, 'tutor_response', 'Good'),
    ({'student_model': 'The student.', 'tutor_response': 'Good!'}, StreamEventEnum.KEY_COMPLETED, 'tutor_response', '!')
]

EXAMPLE_LIST_STR_DICT = [{'student_model': 'The student.', 'tutor_response': 'Good!' }, {'student_model': 'The student.', 'tutor_response': 'Good!' }]

EXAMPLE_LIST_STR_DICT_STREAM_DATA = [
    ([], StreamEventEnum.OBJECT_CREATED, None, None),
    ([{'student_model': 'The', 'tutor_response': ''}], StreamEventEnum.KEY_UPDATED, (0, 'student_model'), 'The'),
    ([{'student_model': 'The student', 'tutor_response': ''}], StreamEventEnum.KEY_UPDATED, (0, 'student_model'), ' student'),
    ([{'student_model': 'The student.', 'tutor_response': ''}], StreamEventEnum.KEY_COMPLETED, (0, 'student_model'), '.'),
    ([{'student_model': 'The student.', 'tutor_response': 'Good'}], StreamEventEnum.KEY_UPDATED, (0, 'tutor_response'), 'Good'),
    ([{'student_model': 'The student.', 'tutor_response': 'Good!'}], StreamEventEnum.KEY_COMPLETED, (0, 'tutor_response'), '!'),
    ([{'student_model': 'The student.', 'tutor_response': 'Good!'}, {'student_model': 'The', 'tutor_response': ''}], StreamEventEnum.KEY_UPDATED, (1, 'student_model'), 'The'),
    ([{'student_model': 'The student.', 'tutor_response': 'Good!'}, {'student_model': 'The student', 'tutor_response': ''}], StreamEventEnum.KEY_UPDATED, (1, 'student_model'), ' student'),
    ([{'student_model': 'The student.', 'tutor_response': 'Good!'}, {'student_model': 'The student.', 'tutor_response': ''}], StreamEventEnum.KEY_COMPLETED, (1, 'student_model'), '.'),
    ([{'student_model': 'The student.', 'tutor_response': 'Good!'}, {'student_model': 'The student.', 'tutor_response': 'Good'}], StreamEventEnum.KEY_UPDATED, (1, 'tutor_response'), 'Good'),
    ([{'student_model': 'The student.', 'tutor_response': 'Good!'}, {'student_model': 'The student.', 'tutor_response': 'Good!'}], StreamEventEnum.KEY_COMPLETED, (1, 'tutor_response'), '!')
]

class ExampleDictSchema(BaseModel):
    student_model: str
    student_correct: bool
    correct_answer_number: int
    tutor_response: str

EXAMPLE_DICT = {'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': 12156, 'tutor_response': 'Good!' }

EXAMPLE_DICT_STREAM_DATA = [
    ({'student_model': '', 'student_correct': None, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.OBJECT_CREATED, None, None),
    ({'student_model': 'The', 'student_correct': None, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.KEY_UPDATED, 'student_model', 'The'),
    ({'student_model': 'The student', 'student_correct': None, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.KEY_UPDATED, 'student_model', ' student'),
    ({'student_model': 'The student.', 'student_correct': None, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.KEY_COMPLETED, 'student_model', '.'),
    ({'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.KEY_COMPLETED, 'student_correct', False),
    ({'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': 12156, 'tutor_response': ''}, StreamEventEnum.KEY_COMPLETED, 'correct_answer_number', 12156),
    ({'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': 12156, 'tutor_response': 'Good'}, StreamEventEnum.KEY_UPDATED, 'tutor_response', 'Good'),
    ({'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': 12156, 'tutor_response': 'Good!'}, StreamEventEnum.KEY_COMPLETED, 'tutor_response', '!')
]

class ExampleStrListSchema(BaseModel):
    list_of_strings: list[str]

EXAMPLE_STR_LIST = {"list_of_strings": ["The student.", "Good!"]}

EXAMPLE_STR_LIST_STREAM_DATA = [
    ({'list_of_strings': []}, StreamEventEnum.OBJECT_CREATED, None, None),
    ({'list_of_strings': ['The']}, StreamEventEnum.KEY_UPDATED, ('list_of_strings', 0), 'The'),
    ({'list_of_strings': ['The student']}, StreamEventEnum.KEY_UPDATED, ('list_of_strings', 0), ' student'),
    ({'list_of_strings': ['The student.']}, StreamEventEnum.KEY_COMPLETED, ('list_of_strings', 0), '.'),
    ({'list_of_strings': ['The student.', 'Good']}, StreamEventEnum.KEY_UPDATED, ('list_of_strings', 1), 'Good'),
    ({'list_of_strings': ['The student.', 'Good!']}, StreamEventEnum.KEY_COMPLETED, ('list_of_strings', 1), '!')
]

class ExampleInnerSchema(BaseModel):
    very_nested: int
    inner_list: list[float]
class ExampleMultiNestedSchema(BaseModel):
    is_this_overengineered: bool
    inner_object: ExampleInnerSchema

EXAMPLE_MULTI_NESTED = {
    'is_this_overengineered' : True,
    'inner_object': {
        'very_nested': 'indeed',
        'inner_list': [123456, -212, 3.14159265],
    }
}

EXAMPLE_MULTI_NESTED_STREAM_DATA = [
    ({'is_this_overengineered': None, 'inner_object': {'very_nested': '', 'inner_list': []}}, StreamEventEnum.OBJECT_CREATED, None, None),
    ({'is_this_overengineered': True, 'inner_object': {'very_nested': '', 'inner_list': []}}, StreamEventEnum.KEY_COMPLETED, 'is_this_overengineered', True),
    ({'is_this_overengineered': True, 'inner_object': {'very_nested': 'inde', 'inner_list': []}}, StreamEventEnum.KEY_UPDATED, ('inner_object', 'very_nested'), 'inde'),
    ({'is_this_overengineered': True, 'inner_object': {'very_nested': 'indeed', 'inner_list': []}}, StreamEventEnum.KEY_COMPLETED, ('inner_object', 'very_nested'), 'ed'),
    ({'is_this_overengineered': True, 'inner_object': {'very_nested': 'indeed', 'inner_list': [123456]}}, StreamEventEnum.KEY_COMPLETED, ('inner_object', 'inner_list', 0), 123456),
    ({'is_this_overengineered': True, 'inner_object': {'very_nested': 'indeed', 'inner_list': [123456, -212]}}, StreamEventEnum.KEY_COMPLETED, ('inner_object', 'inner_list', 1), -212),
    ({'is_this_overengineered': True, 'inner_object': {'very_nested': 'indeed', 'inner_list': [123456, -212, 3.14159265]}}, StreamEventEnum.KEY_COMPLETED, ('inner_object', 'inner_list', 2), 3.14159265)
]


@pytest.mark.parametrize(
    "full_object,expected_stream_data",
    [
        (EXAMPLE_STR_DICT, EXAMPLE_STR_DICT_STREAM_DATA),
        (EXAMPLE_LIST_STR_DICT, EXAMPLE_LIST_STR_DICT_STREAM_DATA),
        (EXAMPLE_DICT, EXAMPLE_DICT_STREAM_DATA),
        (EXAMPLE_STR_LIST, EXAMPLE_STR_LIST_STREAM_DATA),
        (EXAMPLE_MULTI_NESTED, EXAMPLE_MULTI_NESTED_STREAM_DATA)
    ]
)
def test_get_expected_stream_objs(full_object, expected_stream_data):
    out = list(_get_expected_stream_partial_objs(full_object))
    assert out == expected_stream_data


MOCK_ASSISTANT_CHUNK = {"id": "chatcmpl-7GWTw9HlmVFOiXyWNBfNKVFzA55yy", "object": "chat.completion.chunk", "created": 1684172464, "model": "gpt-4-0314", "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}]}
MOCK_FINISH_REASON_CHUNK = {"id": "chatcmpl-7GWTw9HlmVFOiXyWNBfNKVFzA55yy", "object": "chat.completion.chunk", "created": 1684172464, "model": "gpt-4-0314", "choices": [{"delta": {}, "index": 0, "finish_reason": "stop" }]}
MOCK_CONTENT_CHUNK = lambda content: {"id": "chatcmpl-7GWTw9HlmVFOiXyWNBfNKVFzA55yy", "object": "chat.completion.chunk", "created": 1684172464, "model": "gpt-4-0314", "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}]}

def _mock_oai_streaming_chunks(full_object, json_indent_level=2, prefix_str=None, postfix_str=None):
    yield MOCK_ASSISTANT_CHUNK

    full_content = f"{prefix_str if prefix_str else ''}{json.dumps(full_object, indent=json_indent_level)}{postfix_str if postfix_str else ''}"
    full_content_tokens = _tokenize(full_content)

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
        (EXAMPLE_MULTI_NESTED, ExampleMultiNestedSchema, EXAMPLE_MULTI_NESTED_STREAM_DATA, False)
    ]
)
async def test_gpt_stream(full_object, schema_typehint, expected_stream_data, should_support):
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
    mocked_oai_raw_responses = _mock_oai_streaming_chunks(full_object)
    async def async_list_to_generator(my_list):
        for item in my_list:
            yield item
    mock_response = async_list_to_generator(mocked_oai_raw_responses)

    # Create the mock
    with patch.object(openai.ChatCompletion, "acreate", return_value=mock_response) as mock_acreate:
        # Make the mock function asynchronous
        mock_acreate.__aenter__.return_value = MagicMock()
        mock_acreate.__aexit__.return_value = MagicMock()
        mock_acreate.__aenter__.return_value.__aenter__ = MagicMock(return_value=mock_response)

        if not should_support:
            with pytest.raises(NotImplementedError):
                streaming_objects = [obj async for obj in model.stream(messages=messages)]
            return True

        # Call the function and pass the expected parameters
        streaming_objects = model.stream(messages=messages)

        idx = 0
        async for stream_obj in streaming_objects:
            expected_partial_obj, expected_event, expected_update_key, expected_value_change = expected_stream_data[idx]
            expected_obj = StreamingObject[schema_typehint](partial_obj=schema_typehint(**expected_partial_obj), event=expected_event, updated_key=expected_update_key, value_change=expected_value_change)
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
            timeout=60,
            stream=True,
            api_key=None,
        )
        

