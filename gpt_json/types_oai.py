from pydantic import BaseModel


class ChatCompletionDelta(BaseModel):
    content: str | None = None
    role: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    delta: ChatCompletionDelta
    finish_reason: str | None = None
    index: int


class ChatCompletionChunk(BaseModel):
    choices: list[ChatCompletionChunkChoice]
    created: int
    id: str
    model: str
    object: str
