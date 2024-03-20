import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel

from gpt_json.transformations import JsonFixEnum, fix_truncated_json

SchemaType = TypeVar("SchemaType", bound=BaseModel)


class StreamEventEnum(Enum):
    OBJECT_CREATED = "OBJECT_CREATED"
    KEY_UPDATED = "KEY_UPDATED"
    KEY_COMPLETED = "KEY_COMPLETED"
    AWAITING_FIRST_KEY = "AWAITING_FIRST_KEY"


@dataclass
class StreamingObject(Generic[SchemaType]):
    updated_key: str | None
    value_change: str | None
    event: StreamEventEnum
    partial_obj: SchemaType

    schema_model: Type[SchemaType] | None = None

    def __class_getitem__(cls, item):
        cls.schema_model = item
        return cls


def _create_schema_from_partial(
    schema_model: Type[SchemaType], partial: dict[str, Any]
):
    """Creates a pydantic model from a dictionary that only partially defines model values.
    Only supports string field types for now.

    TODO: this is hacky. ideally we want pydantic to implement Partial[SchemaType]
    https://github.com/pydantic/pydantic/issues/1673
    my fix is to create the schema object with all string values for now"""
    cleaned_obj_data = {field: "" for field, typ in schema_model.model_fields.items()}
    cleaned_obj_data.update({k: v for k, v in partial.items() if v is not None})
    return schema_model(**cleaned_obj_data)


def prepare_streaming_object(
    schema_model: Type[SchemaType],
    current_partial_raw: dict[str, Any],
    previous_partial: StreamingObject[SchemaType] | None,
    proposed_event: StreamEventEnum,
) -> StreamingObject[SchemaType]:
    """Prepares a StreamingObject for the next iteration of the stream generator

    :param schema_model: The pydantic model for the full schema being streamed
    :param current_partial_raw: The "raw" JSON object parsed after fixing the partially streamed JSON string
    :param previous_partial: The previous StreamingObject returned by the stream generator
    :param proposed_event: The streaming event "proposed" while fixing the partially streamed JSON string.
    NOTE: this is not necessarily the "official" event returned by this function in the StreamingObject.

    :return: StreamingObject[SchemaType].
    """
    # compute which key was most recently updated
    raw_recent_key = (
        list(current_partial_raw.keys())[-1] if current_partial_raw else None
    )
    updated_key = (
        raw_recent_key if raw_recent_key in schema_model.model_fields else None
    )

    event = proposed_event
    if proposed_event == StreamEventEnum.KEY_UPDATED and updated_key is None:
        # when the updated key is None, we haven't fully streamed the object's first key yet
        event = StreamEventEnum.AWAITING_FIRST_KEY

    partial_obj = _create_schema_from_partial(schema_model, current_partial_raw)

    # compute value update if relevant
    value_change = None
    if event in [StreamEventEnum.KEY_UPDATED, StreamEventEnum.KEY_COMPLETED]:
        prev_value = (
            previous_partial.partial_obj.model_dump()[updated_key]
            if previous_partial is not None and updated_key is not None
            else ""
        )
        curr_value = partial_obj.model_dump().get(updated_key, "")
        if isinstance(prev_value, str) and isinstance(curr_value, str):
            value_change = curr_value.replace(prev_value, "")
        else:
            value_change = curr_value

    return StreamingObject[SchemaType](
        value_change=value_change,
        updated_key=updated_key,
        event=event,
        partial_obj=partial_obj,
    )


def parse_streamed_json(substring: str) -> tuple[dict, StreamEventEnum]:
    fixed_json_str, fix_reason = fix_truncated_json(substring)

    event = StreamEventEnum.KEY_UPDATED
    if fix_reason == JsonFixEnum.UNCLOSED_OBJECT:
        if fixed_json_str.count('"') == 0:
            event = StreamEventEnum.OBJECT_CREATED
        else:
            event = StreamEventEnum.KEY_COMPLETED

    return json.loads(fixed_json_str), event
