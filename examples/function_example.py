import asyncio
from enum import Enum
from json import dumps as json_dumps
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")


class UnitType(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class GetCurrentWeatherRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: UnitType | None = None


class DataPayload(BaseModel):
    data: str


def get_current_weather(request: GetCurrentWeatherRequest):
    """
    Get the current weather in a given location

    The rest of the docstring should be omitted.
    """
    weather_info = {
        "location": request.location,
        "temperature": "72",
        "unit": request.unit,
        "forecast": ["sunny", "windy"],
    }
    return json_dumps(weather_info)


async def runner():
    gpt_json = GPTJSON[DataPayload](API_KEY, functions=[get_current_weather])
    response = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.USER,
                content="What's the weather like in Boston, in F?",
            ),
        ],
    )

    print(response)
    assert response.function_call == get_current_weather
    assert response.function_arg == GetCurrentWeatherRequest(
        location="Boston", unit=UnitType.FAHRENHEIT
    )


asyncio.run(runner())
