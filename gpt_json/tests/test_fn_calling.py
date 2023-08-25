from typing import Callable, Optional, Union

import pytest

from gpt_json.fn_calling import get_base_type, get_function_description, parse_function
from gpt_json.tests.shared import (
    UnitType,
    get_current_weather,
    get_current_weather_async,
    get_weather_additional_args,
    get_weather_no_pydantic,
)


def multi_line_description_fn():
    """
    Test
    description

    Hidden
    description
    """


def single_line_description_fn():
    """Test description"""


@pytest.mark.parametrize(
    "incorrect_fn",
    [
        get_weather_additional_args,
        get_weather_no_pydantic,
    ],
)
def test_parse_function_incorrect_args(incorrect_fn: Callable):
    with pytest.raises(ValueError):
        parse_function(incorrect_fn)


def test_get_base_type():
    assert get_base_type(UnitType | None) == UnitType
    assert get_base_type(Optional[UnitType]) == UnitType
    assert get_base_type(Union[UnitType, None]) == UnitType


def test_get_function_description():
    assert get_function_description(multi_line_description_fn) == "Test description"
    assert get_function_description(single_line_description_fn) == "Test description"


@pytest.mark.parametrize(
    "function,expected_name",
    [
        (get_current_weather, "get_current_weather"),
        (get_current_weather_async, "get_current_weather_async"),
    ],
)
def test_parse_function(function, expected_name: str):
    """
    Assert the formatted schema conforms to the expected JSON-Schema / GPT format.
    """
    assert parse_function(function) == {
        "name": expected_name,
        "description": "Test description",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "title": "Location",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "anyOf": [
                        {
                            "enum": ["celsius", "fahrenheit"],
                            "title": "UnitType",
                            "type": "string",
                        },
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["location"],
        },
    }
