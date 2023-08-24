from typing import Callable, Optional, Union

import pytest

from gpt_json.fn_calling import get_base_type, parse_function
from gpt_json.tests.shared import (
    UnitType,
    get_current_weather,
    get_weather_additional_args,
    get_weather_no_pydantic,
)


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


def test_parse_function():
    """
    Assert the formatted schema conforms to the expected JSON-Schema / GPT format.
    """
    parse_function(get_current_weather) == {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location. Second line should also be included.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
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
