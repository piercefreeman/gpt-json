from re import sub

import pytest

from gpt_json.models import GPTMessage, GPTMessageRole
from gpt_json.prompts import generate_schema_prompt, messages_to_claude_prompt
from gpt_json.tests.shared import MySchema


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
            list[MySchema],
            """
            [
            {{
            "text": str,
            "items": str[],
            "numerical": int | float,
            "sub_element": {{
                "name": str
            }},
            "reason": bool // Explanation
            }}, // Repeat for as many objects as are relevant
            ]
            """,
        ),
    ],
)
def test_generate_schema_prompt(schema_definition, expected: str):
    assert strip_whitespace(
        generate_schema_prompt(schema_definition)
    ) == strip_whitespace(expected)


def test_claude_prompt_builder():
    assert (
        messages_to_claude_prompt(
            [
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content="[HUMAN_TEXT]",
                ),
                GPTMessage(
                    role=GPTMessageRole.ASSISTANT,
                    content="[ASSISTANT_TEXT]",
                ),
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content="[MORE HUMAN TEXT]",
                ),
            ]
        )
        == "\n\nHuman: [HUMAN_TEXT]\n\nAssistant: [ASSISTANT_TEXT]\n\nHuman: [MORE HUMAN TEXT]\n\nAssistant:"
    )


def test_claude_prompt_builder_fail_if_system_message():
    with pytest.raises(
        ValueError, match=".* only support User and Assistant messages.*"
    ):
        messages_to_claude_prompt(
            [
                GPTMessage(
                    role=GPTMessageRole.SYSTEM,
                    content="[SYSTEM_TEXT]",
                ),
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content="[HUMAN TEXT]",
                ),
            ]
        )


def test_claude_prompt_builder_fail_if_incorrect_order():
    with pytest.raises(ValueError, match=".* end with a User message.*"):
        messages_to_claude_prompt(
            [
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content="[HUMAN_TEXT]",
                ),
                GPTMessage(
                    role=GPTMessageRole.ASSISTANT,
                    content="[ASSISTANT_TEXT]",
                ),
            ]
        )

    with pytest.raises(ValueError, match=".* alternate User/Assistant .*"):
        messages_to_claude_prompt(
            [
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content="[HUMAN_TEXT]",
                ),
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content="[MORE HUMAN TEXT]",
                ),
                GPTMessage(
                    role=GPTMessageRole.ASSISTANT,
                    content="[ASSISTANT_TEXT]",
                ),
            ]
        )

    with pytest.raises(ValueError, match=".* first message to be a User message .*"):
        messages_to_claude_prompt(
            [
                GPTMessage(
                    role=GPTMessageRole.ASSISTANT,
                    content="[ASSISTANT_TEXT]",
                ),
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content="[HUMAN_TEXT]",
                ),
            ]
        )
