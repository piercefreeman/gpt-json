from typing import Dict, List, Optional, Union


class ChatCompletionDelta:
    def __init__(
        self,
        content: str | None,
        role: str | None
    ):
        self.content = content
        self.role = role

class ChatCompletionChunkChoice:
    def __init__(
        self,
        delta: Dict[str, str],
        finish_reason: Optional[str],
        index: int,
    ):
        self.delta = delta
        self.finish_reason = finish_reason
        self.index = index

class ChatCompletionChunk:
    def __init__(
        self,
        choices: List[ChatCompletionChunkChoice],
        created: int,
        id: str,
        model: str,
        object: str,
    ):
        self.choices = choices
        self.created = created
        self.id = id
        self.model = model
        self.object = object

    @classmethod
    def from_dict(cls, data: Dict[str, Union[List[Dict[str, Union[Dict[str, str], None, int]]], int, str]]) -> "ChatCompletionChunk":
        choices_data = data["choices"]
        choices = [
            ChatCompletionChunkChoice(
                delta=ChatCompletionDelta(
                    content=choice_data["delta"].get("content"),
                    role=choice_data["delta"].get("role"),
                ),
                finish_reason=choice_data["finish_reason"],
                index=choice_data["index"],
            )
            for choice_data in choices_data
        ]

        return cls(
            choices=choices,
            created=data["created"],
            id=data["id"],
            model=data["model"],
            object=data["object"],
        )