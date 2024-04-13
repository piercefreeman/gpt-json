import random

import pytest
from pydantic import BaseModel

from gpt_json.gpt import GPTJSON
from gpt_json.models import (
    GPTMessage,
    GPTMessageRole,
    GPTModelVersion,
    TextContent,
    TruncationOptions,
    VariableTruncationMode,
)
from gpt_json.truncation import truncate_tokens


def test_fill_messages_truncated():
    class TestSchema(BaseModel):
        summary: str

    gpt = GPTJSON[TestSchema](api_key="TEST")
    assert gpt.fill_messages(
        [
            GPTMessage(
                role=GPTMessageRole.SYSTEM, content=[TextContent(text="system")]
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=[TextContent(text="some long text: {long_text}")],
            ),
        ],
        dict(
            long_text="hello world goodbye world",
        ),
        truncation_options=TruncationOptions(
            target_variable="long_text",
            max_prompt_tokens=8,
            truncation_mode=VariableTruncationMode.BEGINNING,
        ),
        max_response_tokens=None,
    ) == [
        GPTMessage(role=GPTMessageRole.SYSTEM, content=[TextContent(text="system")]),
        GPTMessage(
            role=GPTMessageRole.USER,
            content=[TextContent(text="some long text: hello world")],
        ),
    ]


def test_fill_messages_truncated_failure_case():
    class TestSchema(BaseModel):
        summary: str

    gpt = GPTJSON[TestSchema](api_key="TEST")

    # this should fail because the max_prompt_tokens is too small
    with pytest.raises(ValueError, match=".* max_prompt_tokens .* too small .*"):
        gpt.fill_messages(
            [
                GPTMessage(
                    role=GPTMessageRole.SYSTEM, content=[TextContent(text="system")]
                ),
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content=[TextContent(text="{long_text}")],
                ),
            ],
            dict(
                long_text="hello world goodbye world",
            ),
            truncation_options=TruncationOptions(
                target_variable="long_text",
                max_prompt_tokens=1,
                truncation_mode=VariableTruncationMode.BEGINNING,
            ),
            max_response_tokens=None,
        )


def test_token_truncation_end_mode():
    assert (
        truncate_tokens(
            "hello world goodbye world world",
            GPTModelVersion.GPT_3_5.value.api_name,
            VariableTruncationMode.BEGINNING,
            2,
        )
        == "hello world"
    )


def test_token_truncation_beginning_mode():
    assert (
        truncate_tokens(
            "hello world world goodbye world",
            GPTModelVersion.GPT_3_5.value.api_name,
            VariableTruncationMode.TRAILING,
            2,
        )
        == " goodbye world"
    )


def test_token_truncation_middle_mode():
    assert (
        truncate_tokens(
            "hello world goodbye world world",
            GPTModelVersion.GPT_3_5.value.api_name,
            VariableTruncationMode.MIDDLE,
            1,
        )
        == " goodbye"
    )

    assert (
        truncate_tokens(
            "hello world goodbye world world",
            GPTModelVersion.GPT_3_5.value.api_name,
            VariableTruncationMode.MIDDLE,
            2,
        )
        == " world goodbye"
    )

    assert (
        truncate_tokens(
            "hello world goodbye world world",
            GPTModelVersion.GPT_3_5.value.api_name,
            VariableTruncationMode.MIDDLE,
            3,
        )
        == " world goodbye world"
    )


def test_token_truncation_random_mode():
    random.seed(1)
    assert (
        truncate_tokens(
            "hello world goodbye world",
            GPTModelVersion.GPT_3_5.value.api_name,
            VariableTruncationMode.RANDOM,
            2,
        )
        == "hello world"
    )


def test_token_truncation_custom_mode():
    def _custom_truncate(text_prev):
        return " | ".join(text_prev.split(" | ")[:-1])

    assert (
        truncate_tokens(
            "hello | world | goodbye | world | world",
            GPTModelVersion.GPT_3_5.value.api_name,
            VariableTruncationMode.CUSTOM,
            3,
            _custom_truncate,
        )
        == "hello | world"
    )
