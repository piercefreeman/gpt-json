import asyncio
import os
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
from gpt_json.models import GPTModelVersion

load_dotenv()
API_KEY = getenv("ANTHROPIC_API_KEY")


class SentimentSchema(BaseModel):
    sentiment: str


PROMPT_TEMPLATE = """
Analyze the sentiment of the given text.

Respond with the following JSON schema:

{json_schema}

Text: I love this product. It's the best thing ever!
"""


async def runner():
    gpt_json = GPTJSON[SentimentSchema](API_KEY, model=GPTModelVersion.CLAUDE_100K)
    response, _ = await gpt_json.run(
        # Anthropic doesn't support system prompts, and only supports a single user message
        messages=[
            GPTMessage(
                role=GPTMessageRole.USER,
                content=PROMPT_TEMPLATE,
            ),
        ]
    )
    print(response)
    print(f"Detected sentiment: {response.sentiment}")


asyncio.run(runner())
