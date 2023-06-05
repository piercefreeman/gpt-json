import asyncio
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
from gpt_json.models import TruncationOptions, VariableTruncationMode

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")


class SummarySchema(BaseModel):
    summary: str


SYSTEM_PROMPT = """
Write a three-sentence summary of the user's text input.

Respond with the following JSON schema:

{json_schema}
"""

USER_PROMPT = """{long_text}"""


async def runner():
    with open("examples/raw_data/long_text.txt", "r") as f:
        long_text = f.read()

    gpt_json = GPTJSON[SummarySchema](API_KEY)
    response, _ = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=USER_PROMPT,
            ),
        ],
        format_variables={
            "long_text": long_text,
        },
        truncation_options=TruncationOptions(
            target_variable="long_text",
            truncation_mode=VariableTruncationMode.BEGINNING,
            max_prompt_tokens=300,
        ),
    )
    print(response)
    print(f"Summary: {response.summary}")


asyncio.run(runner())
