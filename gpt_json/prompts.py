from types import UnionType
from typing import List, Type, get_args, get_origin

import anthropic  # type: ignore
from pydantic import BaseModel

from gpt_json.models import GPTMessage, GPTMessageRole


def generate_schema_prompt(schema: Type[BaseModel] | list[Type[BaseModel]]) -> str:
    """
    Converts the pydantic schema into a text representation that can be embedded
    into the prompt payload.

    """

    def generate_payload(model):
        payload = []
        for key, value in model.__fields__.items():
            field_annotation = model.__annotations__[key]
            annotation_origin = get_origin(field_annotation)
            annotation_arguments = get_args(field_annotation)

            if annotation_origin in {list, List}:
                payload.append(f'"{key}": {annotation_arguments[0].__name__}[]')
            elif annotation_origin == UnionType:
                payload.append(
                    f'"{key}": {" | ".join([arg.__name__.lower() for arg in annotation_arguments])}'
                )
            elif issubclass(value.type_, BaseModel):
                payload.append(f'"{key}": {generate_payload(value.type_)}')
            else:
                payload.append(f'"{key}": {value.type_.__name__.lower()}')
            if value.field_info.description:
                payload[-1] += f" // {value.field_info.description}"
        # All brackets are double defined so they will passthrough a call to `.format()` where we
        # pass custom variables
        return "{{\n" + ",\n".join(payload) + "\n}}"

    origin = get_origin(schema)
    args = get_args(schema)

    if origin == list:
        if len(args) > 1:
            raise ValueError("Only one list schema is supported at this time")

        return (
            "["
            + "\n".join(
                [
                    generate_payload(sub_schema)
                    + ", // Repeat for as many objects as are relevant"
                    for sub_schema in args
                ]
            )
            + "]"
        )
    else:
        return generate_payload(schema)


def messages_to_claude_prompt(messages: list[GPTMessage]) -> str:
    """formatting details here: https://console.anthropic.com/docs/troubleshooting/checklist"""
    if any(
        [
            m.role not in [GPTMessageRole.USER, GPTMessageRole.ASSISTANT]
            for m in messages
        ]
    ):
        raise ValueError("CLAUDE models only support User and Assistant messages.")
    if [m.role for m in messages] != [
        [GPTMessageRole.USER, GPTMessageRole.ASSISTANT][idx % 2]
        for idx in range(len(messages))
    ]:
        raise ValueError(
            "CLAUDE models require the first message to be a User message and alternate User/Assistant from there."
        )
    if messages[-1].role != GPTMessageRole.USER:
        raise ValueError(
            "CLAUDE models require specified messages to end with a User message."
        )

    gpt_role_to_prefix = {
        GPTMessageRole.USER.value: anthropic.HUMAN_PROMPT,
        GPTMessageRole.ASSISTANT.value: anthropic.AI_PROMPT,
    }
    base_prompt = "".join(
        [
            f"{gpt_role_to_prefix[message.role.value]} {message.content}"
            for message in messages
        ]
    )
    return base_prompt + anthropic.AI_PROMPT
