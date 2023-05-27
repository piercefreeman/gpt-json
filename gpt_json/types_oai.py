from pydantic import BaseModel


class ChatCompletionDelta(BaseModel):
    content: str | None
    role: str | None


class ChatCompletionChunkChoice(BaseModel):
    delta: ChatCompletionDelta
    finish_reason: str | None
    index: int


class ChatCompletionChunk(BaseModel):
    choices: list[ChatCompletionChunkChoice]
    created: int
    id: str
    model: str
    object: str
