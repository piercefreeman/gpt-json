from types import UnionType
from typing import List, Type, get_args, get_origin

from pydantic import BaseModel

from gpt_json.common import get_field_description, get_model_fields


def generate_schema_prompt(schema: Type[BaseModel]) -> str:
    """
    Converts the pydantic schema into a text representation that can be embedded
    into the prompt payload.

    """

    def generate_payload(model: Type[BaseModel]):
        payload = []
        for key, value in get_model_fields(model).items():
            field_annotation = value.annotation
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
            elif issubclass(field_annotation, BaseModel):
                payload.append(f'"{key}": {generate_payload(field_annotation)}')
            else:
                payload.append(f'"{key}": {field_annotation.__name__.lower()}')
            if get_field_description(value):
                payload[-1] += f" // {get_field_description(value)}"
        # All brackets are double defined so they will passthrough a call to `.format()` where we
        # pass custom variables
        return "{{\n" + ",\n".join(payload) + "\n}}"

    return generate_payload(schema)
