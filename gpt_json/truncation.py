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


def truncate_tokens(text: str, mode: VariableTruncationMode, max_tokens: int):
    truncation_iter = TokenTruncationIterator(text, mode)
    for truncated_text, num_tokens in truncation_iter:
        if num_tokens <= max_tokens:
            return truncated_text


class TokenTruncationIterator:
    def __init__(self, text: str, mode: VariableTruncationMode):
        self.text = text
        self.mode = mode

        self.tokens = oai_approx_tokenize(text)

        self._idx = 0

    def __len__(self):
        # + 1 because we include both full and empty text
        return len(self.tokens) + 1

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration

        if self.mode == VariableTruncationMode.END:
            truncated_text = (
                oai_decode(self.tokens[: -self._idx]) if self._idx else self.text
            )
            num_tokens = len(self.tokens) - self._idx
        elif self.mode == VariableTruncationMode.BEGINNING:
            truncated_text = oai_decode(self.tokens[self._idx :])
            num_tokens = len(self.tokens) - self._idx

        self._idx += 1
        return truncated_text, num_tokens


if __name__ == "__main__":
    print(oai_approx_tokenize("hello world"))
    print(oai_decode(oai_approx_tokenize("hello world")))

    tti = TokenTruncationIterator(
        "hello world goodbye world", VariableTruncationMode.END
    )
    print([x for x in tti])
    tti = TokenTruncationIterator(
        "hello world goodbye world", VariableTruncationMode.BEGINNING
    )
    print([x for x in tti])
