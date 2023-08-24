import asyncio
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")


class SentimentSchema(BaseModel):
    sentiment: int = Field(description="Either -1, 0, or 1.")


SYSTEM_PROMPT = """
Analyze the sentiment of the given text.

Respond with the following JSON schema:

{json_schema}
"""


async def runner():
    gpt_json = GPTJSON[SentimentSchema](API_KEY)
    response = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content="Text: I love this product. It's the best thing ever!",
            ),
        ]
    )
    print(response.response)
    print(f"Detected sentiment: {response.response.sentiment}")


asyncio.run(runner())
