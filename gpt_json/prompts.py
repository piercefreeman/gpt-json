from types import UnionType
from typing import List, get_args, get_origin

from pydantic import BaseModel


def generate_schema_prompt(schema: BaseModel | list[BaseModel]) -> str:
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
                payload.append(f'"{key}": {" | ".join([arg.__name__.lower() for arg in annotation_arguments])}')
            elif issubclass(value.type_, BaseModel):
                payload.append(f'"{key}": {generate_payload(value.type_)}')
            else:
                payload.append(f'"{key}": {value.type_.__name__.lower()}')
            if value.field_info.description:
                payload[-1] += f' // {value.field_info.description}'
        return '{' + ', '.join(payload) + '}'

    origin = get_origin(schema)
    args = get_args(schema)

    if origin == list:
        if len(args) > 1:
            raise ValueError("Only one list schema is supported at this time")

        return '[' + '\n'.join(
            [
                generate_payload(sub_schema) + ", // Repeat for as many objects as are relevant"
                for sub_schema in args
            ]
        ) + ']'
    else:
        return generate_payload(schema)
