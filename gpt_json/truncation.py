import random
from typing import Callable

import tiktoken

from gpt_json.models import GPTModelVersion, VariableTruncationMode


def tokenize(text: str, model: GPTModelVersion) -> list[int]:
    enc = tiktoken.encoding_for_model(model)
    return [tok for tok in enc.encode(text)]


def decode(tokens: list[int], model: GPTModelVersion) -> str:
    enc = tiktoken.encoding_for_model(model)
    return enc.decode(tokens)


def num_tokens_from_messages(
    messages: list[dict[str, str]], model: GPTModelVersion
) -> int:
    """Returns the number of tokens used by a list of messages.
    NOTE: in the future, there may be structural changes to how messages are converted into content
    that affect the number of tokens. More here: https://platform.openai.com/docs/guides/chat/managing-tokens
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


def truncate_tokens(
    text: str,
    model: GPTModelVersion,
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
    elif mode == VariableTruncationMode.CUSTOM and custom_truncate_next is not None:
        tokens = _tokenize(text)
        while len(tokens) > max_tokens:
            tokens = _tokenize(custom_truncate_next(_decode(tokens)))
        return _decode(tokens)


if __name__ == "__main__":
    model_versions = [
        GPTModelVersion.GPT_3_5,
        GPTModelVersion.GPT_4,
    ]
    for model_version in model_versions:
        text = "Hello, my name is John. I am a human."
        tokens = tokenize(text, model_version)
        decoded = decode(tokens, model_version)
        print(f"Model: {model_version}")
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")
        print()
