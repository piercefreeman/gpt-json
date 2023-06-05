from dataclasses import Field

import pytest
from pydantic import BaseModel

from gpt_json.gpt import GPTJSON
from gpt_json.models import (
    GPTMessage,
    GPTMessageRole,
    TruncationOptions,
    VariableTruncationMode,
)
from gpt_json.truncation import TokenTruncationIterator


def test_fill_messages_truncated():
    class TestSchema(BaseModel):
        summary: str

    gpt = GPTJSON[TestSchema](None)
    assert gpt.fill_messages(
        [
            GPTMessage(role=GPTMessageRole.SYSTEM, content="system"),
            GPTMessage(
                role=GPTMessageRole.USER,
                content="some long text: {long_text}",
            ),
        ],
        dict(
            long_text="hello world goodbye world",
        ),
        truncation_options=TruncationOptions(
            target_variable="long_text",
            max_prompt_tokens=20,
            truncation_mode=VariableTruncationMode.END,
        ),
        max_response_tokens=None,
    ) == [
        GPTMessage(role=GPTMessageRole.SYSTEM, content="system"),
        GPTMessage(role=GPTMessageRole.USER, content="some long text: hello world"),
    ]


def test_fill_messages_truncated_failure_case():
    class TestSchema(BaseModel):
        summary: str

    gpt = GPTJSON[TestSchema](None)

    # this should fail because the max_prompt_tokens is too small
    with pytest.raises(ValueError):
        gpt.fill_messages(
            [
                GPTMessage(role=GPTMessageRole.SYSTEM, content="system"),
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content="{long_text}",
                ),
            ],
            dict(
                long_text="hello world goodbye world",
            ),
            truncation_options=TruncationOptions(
                target_variable="long_text",
                max_prompt_tokens=2,
                truncation_mode=VariableTruncationMode.END,
            ),
            max_response_tokens=None,
        )


def test_token_truncation_iterator_end_mode():
    truncation_iterator = TokenTruncationIterator(
        "hello world goodbye world",
        VariableTruncationMode.END,
    )
    assert list(truncation_iterator) == [
        ("hello world goodbye world", 4),
        ("hello world goodbye", 3),
        ("hello world", 2),
        ("hello", 1),
        ("", 0),
    ]


def test_token_truncation_iterator_beginning_mode():
    truncation_iterator = TokenTruncationIterator(
        "hello world goodbye world",
        VariableTruncationMode.BEGINNING,
    )
    assert list(truncation_iterator) == [
        ("hello world goodbye world", 4),
        (" world goodbye world", 3),
        (" goodbye world", 2),
        (" world", 1),
        ("", 0),
    ]
