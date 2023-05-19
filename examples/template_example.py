import asyncio
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")

class QuoteSchema(BaseModel):
    quotes: list[str] = Field(description="Max quantity {max_items}.")

SYSTEM_PROMPT = """
Generate fictitious quotes that are {sentiment}.

{json_schema}
"""

async def runner():
    gpt_json = GPTJSON[QuoteSchema](API_KEY)
    response, _ = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            ),
        ],
        format_variables={"sentiment": "happy", "max_items": 5},
    )

    print(response)
    print(f"Quotes: {response.quotes}")

asyncio.run(runner())
