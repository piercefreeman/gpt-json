from enum import Enum

from pydantic import BaseModel, Field


class MySubSchema(BaseModel):
    name: str


class MySchema(BaseModel):
    text: str
    items: list[str]
    numerical: int | float
    sub_element: MySubSchema
    reason: bool = Field(description="Explanation")


class UnitType(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class GetCurrentWeatherRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: UnitType | None = None


def get_current_weather(request: GetCurrentWeatherRequest):
    """
    Get the current weather in a given location

    The rest of the docstring should be omitted.
    """


def get_weather_additional_args(request: GetCurrentWeatherRequest, other_args: str):
    pass


def get_weather_no_pydantic(other_args: str):
    pass
