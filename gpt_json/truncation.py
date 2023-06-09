import random
from typing import Callable

import anthropic  # type: ignore
import tiktoken

from gpt_json.models import (
    GPTMessage,
    GPTMessageRole,
    GPTModelVersion,
    ModelProvider,
    VariableTruncationMode,
)
from gpt_json.prompts import messages_to_claude_prompt


def tokenize(text: str, model: str) -> list[int]:
    model_provider = ModelProvider.get_provider(model)
    if model_provider == ModelProvider.OPENAI:
        enc = tiktoken.encoding_for_model(model)
        return [tok for tok in enc.encode(text)]
    elif model_provider == ModelProvider.ANTHROPIC:
        enc = anthropic.get_tokenizer()
        return [tok for tok in enc.encode(text).ids]  # type: ignore
    else:
        raise ValueError(f"Unknown model {model}")


def decode(tokens: list[int], model: str) -> str:
    model_provider = ModelProvider.get_provider(model)
    if model_provider == ModelProvider.OPENAI:
        enc = tiktoken.encoding_for_model(model)
        return enc.decode(tokens)
    elif model_provider == ModelProvider.ANTHROPIC:
        enc = anthropic.get_tokenizer()
        return enc.decode(tokens)
    else:
        raise ValueError(f"Unknown model {model}")


def gpt_message_markup_v1(messages: list[dict[str, str]], model: str) -> int:
    """Converts a list of messages into the number of tokens used by the model, following the
    markup rules for GPT-3.5 and GPT-4 defined here: https://platform.openai.com/docs/guides/chat/managing-tokens.
    """
    encoding = tiktoken.encoding_for_model(model)

    num_tokens = 0
    for message in messages:
        num_tokens += (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        )
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def claude_message_markup_v1(messages: list[dict[str, str]], model: str) -> int:
    """Converts a list of messages into the number of tokens used by the model, following the
    markup rules for Anthropic's Claude models: https://console.anthropic.com/docs/troubleshooting/checklist.
    """
    if not messages:
        return 0

    gpt_messages = [
        GPTMessage(role=GPTMessageRole(message["role"]), content=message["content"])
        for message in messages
    ]
    return anthropic.count_tokens(messages_to_claude_prompt(gpt_messages))


MODEL_MESSAGE_MARKUP = {
    GPTModelVersion.GPT_4.value: gpt_message_markup_v1,
    GPTModelVersion.GPT_3_5.value: gpt_message_markup_v1,
    GPTModelVersion.CLAUDE.value: claude_message_markup_v1,
    GPTModelVersion.CLAUDE_100K.value: claude_message_markup_v1,
}


def num_tokens_from_messages(messages: list[dict[str, str]], model: str) -> int:
    """Returns the number of tokens used by a list of messages.
    NOTE: for future models, there may be structural changes to how messages are converted into content
    that affect the number of tokens. Future models should be added to MODEL_MESSAGE_MARKUP.
    """
    if model not in MODEL_MESSAGE_MARKUP:
        raise NotImplementedError(f"Model {model} message markup not implemented")
    return MODEL_MESSAGE_MARKUP[model](messages, model)


def truncate_tokens(
    text: str,
    model: str,
    mode: VariableTruncationMode,
    max_tokens: int,
    custom_truncate_next: Callable[[str], str] | None = None,
) -> str:
    if mode == VariableTruncationMode.CUSTOM and custom_truncate_next is None:
        raise ValueError(
            "VariableTruncationMode.CUSTOM requires a custom_truncate_next function"
        )

    _tokenize = lambda text: tokenize(text, model)
    _decode = lambda tokens: decode(tokens, model)
    if mode == VariableTruncationMode.TRAILING:
        return _decode(_tokenize(text)[-max_tokens:]) if max_tokens else text
    elif mode == VariableTruncationMode.BEGINNING:
        return _decode(_tokenize(text)[:max_tokens])
    elif mode == VariableTruncationMode.MIDDLE:
        middle = len(_tokenize(text)) // 2
        start = middle - max_tokens // 2
        return _decode(_tokenize(text)[start : start + max_tokens])
    elif mode == VariableTruncationMode.RANDOM:
        slice_start = random.randint(0, len(_tokenize(text)) - max_tokens)
        return _decode(_tokenize(text)[slice_start : slice_start + max_tokens])
    elif mode == VariableTruncationMode.CUSTOM:
        tokens = _tokenize(text)
        while len(tokens) > max_tokens:
            tokens = _tokenize(custom_truncate_next(_decode(tokens)))  # type: ignore
        return _decode(tokens)
    else:
        raise ValueError(f"Invalid truncation mode: {mode}")
