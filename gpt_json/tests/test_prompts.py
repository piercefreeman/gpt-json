from re import sub

import pytest

from gpt_json.gpt import ListResponse
from gpt_json.prompts import generate_schema_prompt
from gpt_json.tests.shared import LiteralSchema, MySchema


def strip_whitespace(input_string: str):
    return sub(r"\s+", "", input_string)


@pytest.mark.parametrize(
    "schema_definition,expected",
    [
        (
            MySchema,
            """
            {{
                "text": str,
                "items": str[],
                "numerical": int | float,
                "sub_element": {{
                    "name": str
                }},
                "reason": bool // Explanation
            }}
            """,
        ),
        (
            ListResponse[MySchema],
            """
            {{
            "items": {{
                "text": str,
                "items": str[],
                "numerical": int | float,
                "sub_element": {{
                    "name": str
                }},
                "reason": bool // Explanation
            }}[] // Repeat for as many objects as are relevant
            }}
            """,
        ),
        (
            LiteralSchema,
            """
            {{
            "work_format": "REMOTE" | "OFFICE" | "ANY" // One of the given values
            }}
            """,
        ),
    ],
)
def test_generate_schema_prompt(schema_definition, expected: str):
    assert strip_whitespace(
        generate_schema_prompt(schema_definition)
    ) == strip_whitespace(expected)
