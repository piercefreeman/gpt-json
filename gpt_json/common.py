"""
Pydantic V1 and V2 compatibility layer. Pydantic V2 has a better API for the type inspection
that we do in GPT-JSON, but we can easily bridge some of the concepts in V1.

"""

from typing import Type

from pydantic import BaseModel


def get_field_description(field):
    if hasattr(field, "description"):
        return field.description
    elif hasattr(field, "field_info"):
        return field.field_info.description
    else:
        raise ValueError(f"Unknown pydantic field class structure: {field}")


def get_model_fields(model: Type[BaseModel]):
    if hasattr(model, "model_fields"):
        return model.model_fields
    elif hasattr(model, "__fields__"):
        return model.__fields__
    else:
        raise ValueError(f"Unknown pydantic field class structure: {model}")


def get_model_field_infos(model: Type[BaseModel]):
    if hasattr(model, "model_fields"):
        return model.model_fields
    elif hasattr(model, "__fields__"):
        return {key: value.field_info for key, value in model.__fields__.items()}  # type: ignore
    else:
        raise ValueError(f"Unknown pydantic field class structure: {model}")
