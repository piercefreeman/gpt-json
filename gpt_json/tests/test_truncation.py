import random

import pytest
from pydantic import BaseModel

from gpt_json.gpt import GPTJSON
from gpt_json.models import (
    GPTMessage,
    GPTMessageRole,
    GPTModelVersion,
    TruncationOptions,
    VariableTruncationMode,
)
from gpt_json.truncation import num_tokens_from_messages, truncate_tokens


@pytest.mark.parametrize("model", [model.value for model in GPTModelVersion])
def test_num_tokens_implemented(model):
    # no need to assert anything specific, just that its implemented for all models
    # i.e. doesn't throw an error
    num_tokens_from_messages([], model)


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
            truncation_mode=VariableTruncationMode.BEGINNING,
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
    with pytest.raises(ValueError, match=".* max_prompt_tokens .* too small .*"):
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
                truncation_mode=VariableTruncationMode.BEGINNING,
            ),
            max_response_tokens=None,
        )


def test_token_truncation_end_mode():
    assert (
        truncate_tokens(
            "hello world goodbye world world",
            GPTModelVersion.GPT_3_5.value,
            VariableTruncationMode.BEGINNING,
            2,
        )
        == "hello world"
    )


def test_token_truncation_beginning_mode():
    assert (
        truncate_tokens(
            "hello world world goodbye world",
            GPTModelVersion.GPT_3_5.value,
            VariableTruncationMode.TRAILING,
            2,
        )
        == " goodbye world"
    )


def test_token_truncation_claude():
    assert (
        truncate_tokens(
            "hello world world goodbye world",
            GPTModelVersion.CLAUDE.value,
            VariableTruncationMode.TRAILING,
            2,
        )
        == " goodbye world"
    )


def test_token_truncation_middle_mode():
    assert (
        truncate_tokens(
            "hello world goodbye world world",
            GPTModelVersion.GPT_3_5.value,
            VariableTruncationMode.MIDDLE,
            1,
        )
        == " goodbye"
    )

    assert (
        truncate_tokens(
            "hello world goodbye world world",
            GPTModelVersion.GPT_3_5.value,
            VariableTruncationMode.MIDDLE,
            2,
        )
        == " world goodbye"
    )

    assert (
        truncate_tokens(
            "hello world goodbye world world",
            GPTModelVersion.GPT_3_5.value,
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
            GPTModelVersion.GPT_3_5.value,
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
            GPTModelVersion.GPT_3_5.value,
            VariableTruncationMode.CUSTOM,
            3,
            _custom_truncate,
        )
        == "hello | world"
    )
