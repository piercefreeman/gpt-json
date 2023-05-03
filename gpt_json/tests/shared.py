from pydantic import BaseModel, Field


class MySubSchema(BaseModel):
    name: str


class MySchema(BaseModel):
    text: str
    items: list[str]
    numerical: int | float
    sub_element: MySubSchema
    reason: bool = Field(description="Explanation")
