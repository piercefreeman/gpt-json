# gpt-json

JSON is a beautiful format. It's both human readable and machine readable, which makes it a great format for structured output of LLMs (after all - LLMs are somewhere in the middle). `gpt-json` is a wrapper around GPT that allows for declarative definition of expected output format when you're trying to parse results into a downstream pipeline.

Specifically it:
- Relies on Pydantic schema definitions and type validations
- Allows for defining both dictionaries and lists
- Includes some lightweight manipulation of the output to remove superfluous context and fix broken json
- Includes retry logic for the most common API failures
- Adds typehinting support for both the API and the output schema

## Getting Started

```bash
pip install gpt-json
```

Here's how to use it to generate a schema for simple tasks:

```python
import asyncio

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
from pydantic import BaseModel

class SentimentSchema(BaseModel):
    sentiment: str

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
            )
        ]
    )
    print(response)
    print(f"Detected sentiment: {response.sentiment}")

asyncio.run(runner())
```

```bash
sentiment='positive'
Detected sentiment: positive
```

The `json_schema` is a special keyword that will be replaced with the schema definition at runtime. You should always include this in your payload to ensure the model knows how to format results. However, you can play around with _where_ to include this schema definition; in the system prompt, in the user prompt, at the beginning, or at the end.

You can either typehint the model to return a BaseSchema back, or to provide a list of Multiple BaseSchema. Both of these work:

```python
gpt_json_single = GPTJSON[SentimentSchema](API_KEY)
gpt_json_single = GPTJSON[list[SentimentSchema]](API_KEY)
```

If you want to get more specific about how you expect the model to populate a field, add hints about the value through the "description" field. This helps the model understand what you're looking for, and will help it generate better results.

```python
from pydantic import BaseModel, Field

class SentimentSchema(BaseModel):
    sentiment: int = Field(description="Either -1, 0, or 1.")
```

```
sentiment=1
Detected sentiment: 1
```

## Other Configurations

The `GPTJSON` class supports other configuration parameters at initialization.

| Parameter                   | Type                   | Description                                                                                                                                                                            |
|-----------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model                       | GPTModelVersion \| str | (default: GPTModelVersion.GPT_4) - For convenience we provide the currently supported GPT model versions in the `GPTModelVersion` enum. You can also pass a string value if you want to use another more specific architecture.                                                                                                                                                       |
| auto_trim                   | bool                   | (default: False) - If your input prompt is too long, perhaps because of dynamic injected content, will automatically truncate the text to create enough room for the model's response. |
| auto_trim_response_overhead | int                    | (default: 0) - If you're using auto_trim, configures the max amount of tokens to allow in the model's response.                                                                        |

## Comparison to Other Libraries

A non-exhaustive list of other libraries that address the same problem. None of them were fully compatible with my deployment (hence this library), but check them out:

[jsonformer](https://github.com/1rgs/jsonformer) - Works with any Huggingface model, whereas `gpt-json` is specifically tailored towards the GPT-X family. GPT doesn't output logit probabilities or allow fixed decoder templating so the same approach can't apply.
