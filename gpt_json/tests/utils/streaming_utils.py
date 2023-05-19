from typing import Any

import tiktoken

from gpt_json.streaming import StreamEventEnum

enc = tiktoken.get_encoding("cl100k_base")


def tokenize(text):
    return [enc.decode([tok]) for tok in enc.encode(text)]


def _tuple_merge(t1, t2):
    t1 = (t1,) if not isinstance(t1, tuple) else t1
    t2 = (t2,) if not isinstance(t2, tuple) else t2

    return t1 + t2


class ExpectedPartialObjectStreamHarness:
    """Calling this  class implements the semantics of the behavior we expect from GPTJSON.stream().
    It is only used for testing and demonstrative purposes.
    """

    def __call__(self, full_obj: Any):
        if isinstance(full_obj, list):
            yield from self.handle_list(full_obj)
        elif isinstance(full_obj, dict):
            yield from self.handle_dict(full_obj)
        elif isinstance(full_obj, str):
            yield from self.handle_string(full_obj)
        else:
            yield None, StreamEventEnum.OBJECT_CREATED, None, None
            yield full_obj, StreamEventEnum.KEY_COMPLETED, None, full_obj

    def handle_list(self, full_obj):
        value_iterators = [self(v) for v in full_obj]
        outer_partial: list[Any] = []
        yield outer_partial.copy(), StreamEventEnum.OBJECT_CREATED, None, None
        for key in range(len(full_obj)):
            for inner_partial, event, inner_key, value_change in value_iterators[key]:
                if event == StreamEventEnum.OBJECT_CREATED:
                    outer_partial.append(inner_partial)
                    continue
                outer_partial[key] = inner_partial
                yield_key = _tuple_merge(len(outer_partial) - 1, inner_key)
                yield_key = len(outer_partial) - 1
                if inner_key is not None:
                    yield_key = _tuple_merge(yield_key, inner_key)
                yield outer_partial.copy(), event, yield_key, value_change

    def handle_dict(self, full_obj):
        value_iterators = {k: self(v) for k, v in full_obj.items()}
        # get initial OBJECT_CREATED values for all keys first
        outer_partial = {k: next(v)[0] for k, v in value_iterators.items()}
        yield outer_partial.copy(), StreamEventEnum.OBJECT_CREATED, None, None
        for key in full_obj.keys():
            for inner_partial, event, inner_key, value_change in value_iterators[key]:
                # note: we've already consumed the first value (OBJECT_CREATED) for each key
                outer_partial[key] = inner_partial
                yield_key = key
                if inner_key is not None:
                    yield_key = _tuple_merge(yield_key, inner_key)
                yield outer_partial.copy(), event, yield_key, value_change

    def handle_string(self, full_obj):
        outer_partial = ""
        yield outer_partial, StreamEventEnum.OBJECT_CREATED, None, None

        tokens = tokenize(full_obj)
        for idx, token in enumerate(tokens):
            outer_partial += token
            event = (
                StreamEventEnum.KEY_COMPLETED
                if idx == len(tokens) - 1
                else StreamEventEnum.KEY_UPDATED
            )
            yield outer_partial, event, None, token
