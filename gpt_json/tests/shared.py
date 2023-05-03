from pydantic import BaseModel, Field

class SubModel(BaseModel):
    name: str


class MySchema(BaseModel):
    text: str
    items: list[str]
    numerical: int | float
    sub_element: SubModel
    reason: bool = Field(description="Explanation")
