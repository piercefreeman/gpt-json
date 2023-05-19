import pytest
from pydantic import BaseModel

from gpt_json.streaming import StreamEventEnum
from gpt_json.tests.utils.streaming_utils import ExpectedPartialObjectStreamHarness


class ExampleStrDictSchema(BaseModel):
    student_model: str
    tutor_response: str


EXAMPLE_STR_DICT = {"student_model": "The student.", "tutor_response": "Good!"}

EXAMPLE_STR_DICT_STREAM_DATA = [
    (
        {"student_model": "", "tutor_response": ""},
        StreamEventEnum.OBJECT_CREATED,
        None,
        None,
    ),
    (
        {"student_model": "The", "tutor_response": ""},
        StreamEventEnum.KEY_UPDATED,
        "student_model",
        "The",
    ),
    (
        {"student_model": "The student", "tutor_response": ""},
        StreamEventEnum.KEY_UPDATED,
        "student_model",
        " student",
    ),
    (
        {"student_model": "The student.", "tutor_response": ""},
        StreamEventEnum.KEY_COMPLETED,
        "student_model",
        ".",
    ),
    (
        {"student_model": "The student.", "tutor_response": "Good"},
        StreamEventEnum.KEY_UPDATED,
        "tutor_response",
        "Good",
    ),
    (
        {"student_model": "The student.", "tutor_response": "Good!"},
        StreamEventEnum.KEY_COMPLETED,
        "tutor_response",
        "!",
    ),
]

EXAMPLE_LIST_STR_DICT = [
    {"student_model": "The student.", "tutor_response": "Good!"},
    {"student_model": "The student.", "tutor_response": "Good!"},
]

EXAMPLE_LIST_STR_DICT_STREAM_DATA = [
    ([], StreamEventEnum.OBJECT_CREATED, None, None),
    (
        [{"student_model": "The", "tutor_response": ""}],
        StreamEventEnum.KEY_UPDATED,
        (0, "student_model"),
        "The",
    ),
    (
        [{"student_model": "The student", "tutor_response": ""}],
        StreamEventEnum.KEY_UPDATED,
        (0, "student_model"),
        " student",
    ),
    (
        [{"student_model": "The student.", "tutor_response": ""}],
        StreamEventEnum.KEY_COMPLETED,
        (0, "student_model"),
        ".",
    ),
    (
        [{"student_model": "The student.", "tutor_response": "Good"}],
        StreamEventEnum.KEY_UPDATED,
        (0, "tutor_response"),
        "Good",
    ),
    (
        [{"student_model": "The student.", "tutor_response": "Good!"}],
        StreamEventEnum.KEY_COMPLETED,
        (0, "tutor_response"),
        "!",
    ),
    (
        [
            {"student_model": "The student.", "tutor_response": "Good!"},
            {"student_model": "The", "tutor_response": ""},
        ],
        StreamEventEnum.KEY_UPDATED,
        (1, "student_model"),
        "The",
    ),
    (
        [
            {"student_model": "The student.", "tutor_response": "Good!"},
            {"student_model": "The student", "tutor_response": ""},
        ],
        StreamEventEnum.KEY_UPDATED,
        (1, "student_model"),
        " student",
    ),
    (
        [
            {"student_model": "The student.", "tutor_response": "Good!"},
            {"student_model": "The student.", "tutor_response": ""},
        ],
        StreamEventEnum.KEY_COMPLETED,
        (1, "student_model"),
        ".",
    ),
    (
        [
            {"student_model": "The student.", "tutor_response": "Good!"},
            {"student_model": "The student.", "tutor_response": "Good"},
        ],
        StreamEventEnum.KEY_UPDATED,
        (1, "tutor_response"),
        "Good",
    ),
    (
        [
            {"student_model": "The student.", "tutor_response": "Good!"},
            {"student_model": "The student.", "tutor_response": "Good!"},
        ],
        StreamEventEnum.KEY_COMPLETED,
        (1, "tutor_response"),
        "!",
    ),
]


class ExampleDictSchema(BaseModel):
    student_model: str
    student_correct: bool
    correct_answer_number: int
    tutor_response: str


EXAMPLE_DICT = {
    "student_model": "The student.",
    "student_correct": False,
    "correct_answer_number": 12156,
    "tutor_response": "Good!",
}

EXAMPLE_DICT_STREAM_DATA = [
    (
        {
            "student_model": "",
            "student_correct": None,
            "correct_answer_number": None,
            "tutor_response": "",
        },
        StreamEventEnum.OBJECT_CREATED,
        None,
        None,
    ),
    (
        {
            "student_model": "The",
            "student_correct": None,
            "correct_answer_number": None,
            "tutor_response": "",
        },
        StreamEventEnum.KEY_UPDATED,
        "student_model",
        "The",
    ),
    (
        {
            "student_model": "The student",
            "student_correct": None,
            "correct_answer_number": None,
            "tutor_response": "",
        },
        StreamEventEnum.KEY_UPDATED,
        "student_model",
        " student",
    ),
    (
        {
            "student_model": "The student.",
            "student_correct": None,
            "correct_answer_number": None,
            "tutor_response": "",
        },
        StreamEventEnum.KEY_COMPLETED,
        "student_model",
        ".",
    ),
    (
        {
            "student_model": "The student.",
            "student_correct": False,
            "correct_answer_number": None,
            "tutor_response": "",
        },
        StreamEventEnum.KEY_COMPLETED,
        "student_correct",
        False,
    ),
    (
        {
            "student_model": "The student.",
            "student_correct": False,
            "correct_answer_number": 12156,
            "tutor_response": "",
        },
        StreamEventEnum.KEY_COMPLETED,
        "correct_answer_number",
        12156,
    ),
    (
        {
            "student_model": "The student.",
            "student_correct": False,
            "correct_answer_number": 12156,
            "tutor_response": "Good",
        },
        StreamEventEnum.KEY_UPDATED,
        "tutor_response",
        "Good",
    ),
    (
        {
            "student_model": "The student.",
            "student_correct": False,
            "correct_answer_number": 12156,
            "tutor_response": "Good!",
        },
        StreamEventEnum.KEY_COMPLETED,
        "tutor_response",
        "!",
    ),
]


class ExampleStrListSchema(BaseModel):
    list_of_strings: list[str]


EXAMPLE_STR_LIST = {"list_of_strings": ["The student.", "Good!"]}

EXAMPLE_STR_LIST_STREAM_DATA = [
    ({"list_of_strings": []}, StreamEventEnum.OBJECT_CREATED, None, None),
    (
        {"list_of_strings": ["The"]},
        StreamEventEnum.KEY_UPDATED,
        ("list_of_strings", 0),
        "The",
    ),
    (
        {"list_of_strings": ["The student"]},
        StreamEventEnum.KEY_UPDATED,
        ("list_of_strings", 0),
        " student",
    ),
    (
        {"list_of_strings": ["The student."]},
        StreamEventEnum.KEY_COMPLETED,
        ("list_of_strings", 0),
        ".",
    ),
    (
        {"list_of_strings": ["The student.", "Good"]},
        StreamEventEnum.KEY_UPDATED,
        ("list_of_strings", 1),
        "Good",
    ),
    (
        {"list_of_strings": ["The student.", "Good!"]},
        StreamEventEnum.KEY_COMPLETED,
        ("list_of_strings", 1),
        "!",
    ),
]


class ExampleInnerSchema(BaseModel):
    very_nested: int
    inner_list: list[float]


class ExampleMultiNestedSchema(BaseModel):
    is_this_overengineered: bool
    inner_object: ExampleInnerSchema


EXAMPLE_MULTI_NESTED = {
    "is_this_overengineered": True,
    "inner_object": {
        "very_nested": "indeed",
        "inner_list": [123456, -212, 3.14159265],
    },
}

EXAMPLE_MULTI_NESTED_STREAM_DATA = [
    (
        {
            "is_this_overengineered": None,
            "inner_object": {"very_nested": "", "inner_list": []},
        },
        StreamEventEnum.OBJECT_CREATED,
        None,
        None,
    ),
    (
        {
            "is_this_overengineered": True,
            "inner_object": {"very_nested": "", "inner_list": []},
        },
        StreamEventEnum.KEY_COMPLETED,
        "is_this_overengineered",
        True,
    ),
    (
        {
            "is_this_overengineered": True,
            "inner_object": {"very_nested": "inde", "inner_list": []},
        },
        StreamEventEnum.KEY_UPDATED,
        ("inner_object", "very_nested"),
        "inde",
    ),
    (
        {
            "is_this_overengineered": True,
            "inner_object": {"very_nested": "indeed", "inner_list": []},
        },
        StreamEventEnum.KEY_COMPLETED,
        ("inner_object", "very_nested"),
        "ed",
    ),
    (
        {
            "is_this_overengineered": True,
            "inner_object": {"very_nested": "indeed", "inner_list": [123456]},
        },
        StreamEventEnum.KEY_COMPLETED,
        ("inner_object", "inner_list", 0),
        123456,
    ),
    (
        {
            "is_this_overengineered": True,
            "inner_object": {"very_nested": "indeed", "inner_list": [123456, -212]},
        },
        StreamEventEnum.KEY_COMPLETED,
        ("inner_object", "inner_list", 1),
        -212,
    ),
    (
        {
            "is_this_overengineered": True,
            "inner_object": {
                "very_nested": "indeed",
                "inner_list": [123456, -212, 3.14159265],
            },
        },
        StreamEventEnum.KEY_COMPLETED,
        ("inner_object", "inner_list", 2),
        3.14159265,
    ),
]


@pytest.mark.parametrize(
    "full_object,expected_stream_data",
    [
        (EXAMPLE_STR_DICT, EXAMPLE_STR_DICT_STREAM_DATA),
        (EXAMPLE_LIST_STR_DICT, EXAMPLE_LIST_STR_DICT_STREAM_DATA),
        (EXAMPLE_DICT, EXAMPLE_DICT_STREAM_DATA),
        (EXAMPLE_STR_LIST, EXAMPLE_STR_LIST_STREAM_DATA),
        (EXAMPLE_MULTI_NESTED, EXAMPLE_MULTI_NESTED_STREAM_DATA),
    ],
)
def test_get_expected_stream_objs(full_object, expected_stream_data):
    out = list(ExpectedPartialObjectStreamHarness()(full_object))
    assert out == expected_stream_data
