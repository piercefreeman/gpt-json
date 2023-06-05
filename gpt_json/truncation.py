import random
from typing import Callable

import tiktoken

from gpt_json.models import VariableTruncationMode

enc = tiktoken.get_encoding("cl100k_base")


def oai_approx_tokenize(text):
    return [tok for tok in enc.encode(text)]


def oai_decode(tokens):
    return enc.decode(tokens)


def approx_num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages.
    NOTE: this is only approximate, as there may be minor differences between models.
    More here: https://platform.openai.com/docs/guides/chat/managing-tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

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
    mode: VariableTruncationMode,
    max_tokens: int,
    custom_truncate_next: Callable[[str], str] | None = None,
):
    if mode == VariableTruncationMode.TRAILING:
        return (
            oai_decode(oai_approx_tokenize(text)[-max_tokens:]) if max_tokens else text
        )
    elif mode == VariableTruncationMode.BEGINNING:
        return oai_decode(oai_approx_tokenize(text)[:max_tokens])
    elif mode == VariableTruncationMode.MIDDLE:
        middle = len(oai_approx_tokenize(text)) // 2
        start = middle - max_tokens // 2
        return oai_decode(oai_approx_tokenize(text)[start : start + max_tokens])
    elif mode == VariableTruncationMode.RANDOM:
        return oai_decode(random.sample(oai_approx_tokenize(text), max_tokens))
    elif mode == VariableTruncationMode.CUSTOM and custom_truncate_next is not None:
        tokens = oai_approx_tokenize(text)
        while len(tokens) > max_tokens:
            tokens = oai_approx_tokenize(custom_truncate_next(oai_decode(tokens)))
        return oai_decode(tokens)
