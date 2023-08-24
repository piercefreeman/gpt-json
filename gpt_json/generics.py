from typing import Any, Type, TypeVar, get_args, get_origin

from pydantic import BaseModel, Field, create_model


def get_typevar_mapping(t: Any) -> dict[TypeVar, Any]:
    origin = get_origin(t)
    if not origin:
        raise ValueError("The input is not an instantiated generic type")

    base_parameters = getattr(origin, "__parameters__", [])
    instantiated_parameters = get_args(t)

    if len(base_parameters) != len(instantiated_parameters):
        raise ValueError(
            "The number of parameters in the base class doesn't match the instantiated class"
        )

    return dict(zip(base_parameters, instantiated_parameters))


def resolve_type(type_: Any, typevar_mapping: dict) -> Any:
    """Resolve a type based on the typevar_mapping."""
    if type_ in typevar_mapping:
        return typevar_mapping[type_]
    elif get_origin(type_) is not None:
        resolved_args = tuple(
            resolve_type(arg, typevar_mapping) for arg in get_args(type_)
        )
        return get_origin(type_)[*resolved_args]
    else:
        return type_


def resolve_generic_model(t: Any) -> Type[BaseModel]:
    """
    Pydantic doesn't natively have support for generic-defined models. This function
    takes in a properly typehinted generic and recursively resolves it to the actual type.

    """
    if get_origin(t) is None:
        # If it's not a generic, return it as is.
        return t
    else:
        # Get a mapping from TypeVars to their actual types
        typevar_mapping = get_typevar_mapping(t)

        base_model = get_origin(t)

        # Create a dict with all the fields from the original model
        fields = {}
        for name, type_ in base_model.__annotations__.items():
            original_field = base_model.model_fields.get(name)
            if original_field:
                fields[name] = (type_, original_field)
            else:
                fields[name] = (type_, Field())

        # Replace the fields that have a TypeVar with their resolved types
        for name, (type_, field) in fields.items():
            resolved_annotation = resolve_type(type_, typevar_mapping)
            fields[name] = (resolved_annotation, field)
            if hasattr(field, "annotation"):
                field.annotation = resolved_annotation

        # Use the Pydantic's create_model function to create a new model with the resolved fields
        return create_model(base_model.__name__, **fields)  # type: ignore
