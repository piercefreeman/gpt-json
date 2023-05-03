from re import sub

import pytest

from gpt_json.prompts import generate_schema_prompt
from gpt_json.tests.shared import MySchema


def strip_whitespace(input_string: str):
    return sub(r"\s+", "", input_string)

@pytest.mark.parametrize(
    "schema_definition,expected",
    [
        (
            MySchema,
            """
            {
                "text": str,
                "items": str[],
                "numerical": int | float,
                "sub_element": {
                    "name": str
                },
                "reason": bool // Explanation
            }
            """
        ),
        (
            list[MySchema],
            """
            [
            {
            "text": str,
            "items": str[],
            "numerical": int | float,
            "sub_element": {
                "name": str
            },
            "reason": bool // Explanation
            }, // Repeat for as many objects as are relevant
            ]
            """
        )
    ]
)
def test_generate_schema_prompt(schema_definition, expected: str):
    assert strip_whitespace(generate_schema_prompt(schema_definition)) == strip_whitespace(expected)
