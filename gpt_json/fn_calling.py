from inspect import getdoc, signature
from types import UnionType
from typing import Any, Callable, Dict, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel


def parse_function(fn: Callable) -> Dict[str, Any]:
    """
    Parse a python function into a JSON schema that can be used by OpenAPI. We use
    the first line of the docstring as the description, and the rest of the docstring
    is ignored. We also assume that functions only have one parameter, which is a Pydantic
    model with the required typehints and description strings.

    API Reference: https://platform.openai.com/docs/api-reference/chat/create

    """
    parameter_type = get_argument_for_function(fn)
    description = get_function_description(fn)

    # Parse the parameter type into a JSON schema
    parameter_schema = model_to_parameter_schema(parameter_type)
    return {
        "name": function_to_name(fn),
        "description": description,
        "parameters": parameter_schema,
    }


def model_to_parameter_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    formatted_json = resolve_refs(model.model_json_schema())
    return {
        "type": "object",
        "properties": formatted_json["properties"],
        "required": formatted_json["required"],
    }


def function_to_name(fn: Callable) -> str:
    return fn.__name__


def get_function_description(fn: Callable) -> str:
    """
    The description of a function is everything before an empty linebreak.

    For instance:

    ```
    A
    B

    C
    ```

    Would return "A B"

    """
    docstring = getdoc(fn) or ""
    lines = docstring.strip().split("\n")
    description_lines = []
    for line in lines:
        if not line.strip():
            break
        description_lines.append(line.strip())
    return " ".join(description_lines)


def get_argument_for_function(fn: Callable) -> Type[BaseModel]:
    """
    Function definitions are expected to have one argument, which is a pydantic BaseModel that captures
    the input parameters and optional descriptions of the field values. This function
    validates that the input function only has that one argument, and returns the type of that argument.

    """
    # Parse the model inputs
    parameters = list(signature(fn).parameters.values())

    # Determine if we only have one parameter
    if len(parameters) != 1:
        raise ValueError(
            f"Only one argument is allowed as the function input: {fn} {parameters}"
        )

    # Get the parameter type
    parameter_type = parameters[0].annotation
    if not issubclass(parameter_type, BaseModel):
        raise ValueError(
            f"Only Pydantic objects are allowed as function inputs: {fn} {parameter_type}"
        )

    return parameter_type


def get_base_type(field_type):
    """
    Given a type annotation that might be a Union or Optional, return the base type.
    For instance, if the type is Union[None, int], return int.

    """
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
    """
    Given a JSON-Schema, replace all $ref references to their definitions.

    When JSON Schemas are exported by pydantic, use of nested fields (like enums) will be defined
    in a separate $defs lookup table. This is a valid OpenAPI schema but makes for a less-obvious
    definition for the GPT API. Additionally, none of the docs use the $defs format, so conceivably
    the model was not fine-tuned on this particular format.

    This function takes a schema complete with $defs and resolves all the references to their
    full definition. The resulting payload is what we send to the GPT API.

    """
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
