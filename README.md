# gpt-json

JSON is a beautiful format. It's human readable and machine readable, which makes it a great format for structured output of LLMs. `gpt-json` is a wrapper around GPT that allows for declarative definition of expected output format.

Specifically it:
- Relies on pydantic schema definitions and type validations
- Includes some lightweight manipulation of the output to remove superfluous context and fix broken json
- Includes retry logic for the most common API failures
- Adds typehinting support for both the API and the output

## Getting Started

```bash
pip install gpt-json
```

## Other Libraries

A non-exhaustive list of other libraries that address the same problem. None of them were fully compatible with my deployment (hence this library), but check them out:

[jsonformer](https://github.com/1rgs/jsonformer) - Works with any Huggingface model. This library is tailored towards the GPT-X family.
