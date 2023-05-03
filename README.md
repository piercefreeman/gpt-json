# gpt-json

JSON is a beautiful format. It's human readable and machine readable, which makes it a great format for structured output of LLMs. This library includes a wrapper around GPT to encourage it to respond with a JSON schema. It attempts to be as non-invasive as possible by mirroring the `openai` library schema with additional typing support.

## Other Libraries

A non-exhaustive list of other libraries that address the same problem. None of them were fully compatible with my deployment (hence this library), but check them out:

[jsonformer](https://github.com/1rgs/jsonformer) - Works with any Huggingface model. This library is tailored towards the GPT-X family.
