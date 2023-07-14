import asyncio
import random
from itertools import islice
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
from gpt_json.models import TruncationOptions, VariableTruncationMode

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")


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
    examples = [ex for ex in islice(train_dataset, 200)]
    random.shuffle(examples)
    print("Done!")

    def render_example(ex) -> str:
        return f"Text: {ex['review_body']}\nSentiment: {'positive' if ex['stars'] > 3 else 'negative'}"

    def render_examples(examples) -> str:
        return "\n".join(render_example(ex) for ex in examples)

    # to truncate a single few-shot example, we define a function that
    # removes the last 2 lines of the few-shot area text
    def few_shot_truncate_next(
        text: str,
    ) -> str:
        return "\n".join(text.split("\n")[:-2])

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
            "examples": render_examples(examples),
        },
        # if you don't truncate, the prompt will be too long
        truncation_options=TruncationOptions(
            target_variable="examples",
            max_prompt_tokens=8000,  # close to max gpt4 can handle
            truncation_mode=VariableTruncationMode.CUSTOM,
            custom_truncate_next=few_shot_truncate_next,
        ),
    )
    print(response)
    print(f"Detected sentiment: {response.sentiment}")


asyncio.run(runner())
