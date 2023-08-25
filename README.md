# gpt-json

`gpt-json` is a wrapper around GPT that allows for declarative definition of expected output format. Set up a schema, write a prompt, and get results back as beautiful typehinted objects.

Specifically this library:
- Utilizes Pydantic schema definitions for type casting and validations
- Adds typehinting for both the API and the output schema
- Allows GPT to respond with both single-objects and lists of objects
- Includes some lightweight transformations of the output to remove superfluous context and fix broken json
- Includes retry logic for the most common API failures
- Formats the JSON schema as a flexible prompt that can be added into any message
- Supports templating of prompts to allow for dynamic content
- Validate typehinted function calls in the new GPT models, to better support agent creation

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
    payload = await gpt_json.run(
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
    print(payload.response)
    print(f"Detected sentiment: {payload.response.sentiment}")

asyncio.run(runner())
```

```bash
sentiment='positive'
Detected sentiment: positive
```

The `json_schema` is a special keyword that will be replaced with the schema definition at runtime. You should always include this in your payload to ensure the model knows how to format results. However, you can play around with _where_ to include this schema definition; in the system prompt, in the user prompt, at the beginning, or at the end.

You can either typehint the model to return a BaseSchema back, or to provide a list of multiple BaseSchema. Both of these work:

```python
from gpt_json.gpt import GPTJSON, ListResponse

gpt_json_single = GPTJSON[SentimentSchema](API_KEY)
gpt_json_multiple = GPTJSON[ListResponse[SentimentSchema]](API_KEY)
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

## Prompt Variables

In addition to the `json_schema` template keyword, you can also add arbitrary variables into your messages. This allows you to more easily insert user generated content or dynamically generate prompts based on the results of previous messages.

```python
class QuoteSchema(BaseModel):
    quotes: list[str]

SYSTEM_PROMPT = """
Generate fictitious quotes that are {sentiment}.

{json_schema}
"""

gpt_json = GPTJSON[QuoteSchema](API_KEY)
response = await gpt_json.run(
    messages=[
        GPTMessage(
            role=GPTMessageRole.SYSTEM,
            content=SYSTEM_PROMPT,
        ),
    ],
    format_variables={"sentiment": "happy"},
)
```

When calling the `.run()` function you can pass it the values that should be filled in this template. This also extends to field descriptions as well, so you can specify custom behavior on a per-field basis.

```python
class QuoteSchema(BaseModel):
    quotes: list[str] = Field(description="Max quantity {max_items}.")

SYSTEM_PROMPT = """
Generate fictitious quotes that are {sentiment}.

{json_schema}
"""

gpt_json = GPTJSON[QuoteSchema](API_KEY)
response = await gpt_json.run(
    messages=[
        GPTMessage(
            role=GPTMessageRole.SYSTEM,
            content=SYSTEM_PROMPT,
        ),
    ],
    format_variables={"sentiment": "happy", "max_items": 5},
)
```

## Function Calls

`gpt-3.5-turbo-0613` and `gpt-4-0613` were fine-tuned to support a specific syntax for function calls. We support this syntax in `gpt-json` as well. Here's an example of how to use it:

```python
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

    assert response.function_call == get_current_weather
    assert response.function_arg == GetCurrentWeatherRequest(
        location="Boston", unit=UnitType.FAHRENHEIT
    )
```

The response provides the original function alongside a formatted Pydantic object. If users want to execute the function, they can run response.function_call(response.function_arg). We will parse the get_current_weather function and the GetCurrentWeatherRequest parameter into the format that GPT expects, so it is more likely to return you a correct function execution.

GPT makes no guarantees about the validity of the returned functions. They could hallucinate a function name or the function signature. To address these cases, the run() function may now throw two new exceptions:

`InvalidFunctionResponse` - The function name is incorrect.
`InvalidFunctionParameters` - The function name is correct, but doesn't match the input schema that was provided.

## Other Configurations

The `GPTJSON` class supports other configuration parameters at initialization.

| Parameter                   | Type                   | Description                                                                                                                                                                            |
|-----------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model                       | GPTModelVersion \| str | (default: GPTModelVersion.GPT_4) - For convenience we provide the currently supported GPT model versions in the `GPTModelVersion` enum. You can also pass a string value if you want to use another more specific architecture.                                                                                                                                                       |
| auto_trim                   | bool                   | (default: False) - If your input prompt is too long, perhaps because of dynamic injected content, will automatically truncate the text to create enough room for the model's response. |
| auto_trim_response_overhead | int                    | (default: 0) - If you're using auto_trim, configures the max amount of tokens to allow in the model's response.                                                                        |
| **kwargs | Any | Any other parameters you want to pass to the underlying `GPT` class, will just be a passthrough. |

## Transformations

GPT (especially GPT-4) is relatively good at formatting responses at JSON, but it's not perfect. Some of the more common issues are:

- *Response truncation*: Since GPT is not internally aware of its response length limit, JSON payloads will sometimes exhaust the available token space. This results in a broken JSON payload where much of the data is valid but the JSON object is not closed, which is not valid syntax. There are many cases where this behavior is actually okay for production applications - for instance, if you list 100 generated strings, it's sometimes okay for you to take the 70 that actually rendered. In this case, `gpt-json` will attempt to fix the truncated payload by recreating the JSON object and closing it.
- *Boolean variables*: GPT will sometimes confuse valid JSON boolean values with the boolean tokens that are used in other languages. The most common is generating `True` instead of `true`. `gpt-json` will attempt to fix these values.

When calling `gpt_json.run()`, we return a tuple of values:

```python
payload = await gpt_json.run(...)

print(transformations.fix_transforms)
```

```bash
FixTransforms(fixed_truncation=True, fixed_bools=False)
```

The first object is your generated Pydantic model. The second object is our correction storage object `FixTransforms`. This dataclass contains flags for each of the supported transformation cases that are sketched out above. This allows you to determine whether the response was explicitly parsed from the GPT JSON, or was passed through some middlelayers to get a correct output. From there you can accept or reject the response based on your own business logic.

*Where you can help*: There are certainly more areas of common (and not-so-common failures). If you see these, please add a test case to the unit tests. If you can write a handler to help solve the general case, please do so. Otherwise flag it as a `pytest.xfail` and we'll add it to the backlog.

## Testing

We use poetry for package management. To run the bundled tests, clone the package from github.

```bash
poetry install
poetry run pytest .
```

Our focus is on making unit tests as robust as possible. The variability with GPT should be in its language model, not in its JSON behavior! This is still certainly a work in progress. If you see an edge case that isn't covered, please add it to the test suite.

## Comparison to Other Libraries

A non-exhaustive list of other libraries that address the same problem. None of them were fully compatible with my deployment (hence this library), but check them out:

[jsonformer](https://github.com/1rgs/jsonformer) - Works with any Huggingface model, whereas `gpt-json` is specifically tailored towards the GPT-X family. GPT doesn't output logit probabilities or allow fixed decoder templating so the same approach can't apply.

## Formatting

We use black and mypy for formatting. You can set up a pre-commit git hook to do this automatically via the `./lint.sh` helper file.

If you perform a bulk reformatting to the codebase, you should add your most recent commit to the `.git-blame-ignore-revs` file and run:

```
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
