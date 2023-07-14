from types import UnionType
from typing import List, Type, get_args, get_origin

from pydantic import BaseModel


def generate_schema_prompt(schema: Type[BaseModel]) -> str:
    """
    Converts the pydantic schema into a text representation that can be embedded
    into the prompt payload.

    """

    def generate_payload(model: Type[BaseModel]):
        payload = []
        for key, value in model.__fields__.items():
            field_annotation = model.__annotations__[key]
            annotation_origin = get_origin(field_annotation)
            annotation_arguments = get_args(field_annotation)

            if annotation_origin in {list, List}:
                if issubclass(annotation_arguments[0], BaseModel):
                    payload.append(
                        f'"{key}": {generate_payload(annotation_arguments[0])}[]'
                    )
                else:
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

    return generate_payload(schema)
