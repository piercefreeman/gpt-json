import asyncio
import gzip
import os
import random
from itertools import islice
from os import getenv

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")


# Usage example


class SentimentSchema(BaseModel):
    sentiment: str


SYSTEM_PROMPT = """
Analyze the sentiment of the given text.

Few-shot examples:
{examples}

Respond with the following JSON schema:

{json_schema}
"""


async def runner():
    print("Downloading few-shot data...")
    from datasets import load_dataset

    dataset = load_dataset("amazon_reviews_multi", "en", streaming=True)
    train_dataset = dataset["train"]
    examples = [ex for ex in islice(train_dataset, 1000)]
    random.shuffle(examples)

    def render_example(ex) -> str:
        return f"Text: {ex['review_body']}\nSentiment: {'positive' if ex['stars'] > 3 else 'negative'}"

    rendered_examples = "\n\n".join(render_example(ex) for ex in examples)

    gpt_json = GPTJSON[SentimentSchema](API_KEY)
    response, _ = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content="Text: I love this product. It's the best thing ever!",
            ),
        ],
        format_variables={
            "examples": examples,
        },
        contextaware_options={
            "target": "examples",
            "elem_render": render_example,
            "elem_separator": "\n\n",
        },
    )
    print(response)
    print(f"Detected sentiment: {response.sentiment}")


asyncio.run(runner())
