from dataclasses import dataclass
from enum import Enum
from inspect import getdoc, signature
from types import UnionType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel


@dataclass
class FunctionResponse:
    caller_content: str
    internal_context: Any


def function_to_name(fn: Callable) -> str:
    return fn.__name__


def get_request_from_function(fn: Callable) -> Type[BaseModel]:
    """
    Parse the pydantic schema that is used as the only argument
    for the functions.

    """
    # Parse the model inputs
    parameters = list(signature(fn).parameters.values())

    # Determine if we only have one parameter
    if len(parameters) != 1:
        raise ValueError("Only one Pydantic object is allowed to LLM function")

    # Get the parameter type
    parameter_type = parameters[0].annotation
    if not issubclass(parameter_type, BaseModel):
        raise ValueError("Only Pydantic objects are allowed to LLM function")

    return parameter_type


def parse_function(fn: Callable) -> Dict[str, Any]:
    """
    Parse a python function into a JSON schema that can be used by OpenAPI. We use
    the first line of the docstring as the description, and the rest of the docstring
    is ignored. We also assume that functions only have one parameter, which is a Pydantic
    model with the required typehints and description strings.

    API Reference: https://platform.openai.com/docs/api-reference/chat/create

    """
    docstring = getdoc(fn) or ""
    lines = docstring.strip().split("\n")
    description = lines[0] if lines else None

    parameter_type = get_request_from_function(fn)

    # Parse the parameter type into a JSON schema
    parameter_schema = model_to_parameter_schema(parameter_type)
    print("parameter_schema", parameter_schema)
    return {
        "name": function_to_name(fn),
        "description": description,
        "parameters": parameter_schema,
    }


def get_base_type(field_type):
    origin = get_origin(field_type)
    args = get_args(field_type)

    if isinstance(field_type, UnionType):
        non_none_types = [t for t in field_type.__args__ if t is not type(None)]
        if len(non_none_types) == 1:
            return non_none_types[0]
        elif len(non_none_types) > 1:
            raise ValueError("Polymorphic types not supported")
    elif origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            # We've got only one non-None type in the Union
            return non_none_args[0]
        elif len(non_none_args) > 1:
            raise ValueError("We don't support polymorphic types")
    elif origin is Optional:
        return args[0]

    return field_type


def resolve_refs(schema, defs=None):
    if defs is None:
        defs = schema.get("$defs", {})

    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_key = schema["$ref"].split("/")[
                -1
            ]  # assuming $ref format is like '#/$defs/UnitType'
            return resolve_refs(defs[ref_key], defs)

        return {k: resolve_refs(v, defs) for k, v in schema.items()}

    if isinstance(schema, list):
        return [resolve_refs(item, defs) for item in schema]

    return schema


def model_to_parameter_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    formatted_json = resolve_refs(model.model_json_schema())
    return {
        "type": "object",
        "properties": formatted_json["properties"],
        "required": formatted_json["required"],
    }
