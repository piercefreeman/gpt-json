import random
from typing import Callable

import tiktoken

from gpt_json.models import VariableTruncationMode


def tokenize(text: str, model: str) -> list[int]:
    enc = tiktoken.encoding_for_model(model)
    return [tok for tok in enc.encode(text)]


def decode(tokens: list[int], model: str) -> str:
    enc = tiktoken.encoding_for_model(model)
    return enc.decode(tokens)


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
