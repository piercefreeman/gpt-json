import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, create_model

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

  schema_model: Type[SchemaType] = None
     
  def __class_getitem__(cls, item):
    new_cls = super().__class_getitem__(item)
    new_cls.schema_model = item
    return new_cls


def prepare_streaming_object(schema_model: SchemaType, curr_partial_raw: dict[str, Any], prev_partial: 'StreamingObject[SchemaType]', proposed_event: StreamEventEnum) -> StreamingObject[SchemaType]:
   # compute which key was most recently updated 
   raw_recent_key = list(curr_partial_raw.keys())[-1] if curr_partial_raw else None
   updated_key = raw_recent_key if raw_recent_key in schema_model.__fields__ else None
   
   event = proposed_event
   if proposed_event == StreamEventEnum.KEY_UPDATED and updated_key is None:
      # when the updated key is None, we haven't fully streamed the object's first key yet 
      event = StreamEventEnum.AWAITING_FIRST_KEY
   
   # TODO: this is hacky. ideally we want pydantic to implement Partial[SchemaType]
   # https://github.com/pydantic/pydantic/issues/1673
   # my fix is to create the schema object with all string values for now
   cleaned_obj_data = {field: "" for field, typ in schema_model.__fields__.items()}
   cleaned_obj_data.update({k:v for k,v in curr_partial_raw.items() if v is not None})
   print(cleaned_obj_data)
   partial_obj = schema_model(**cleaned_obj_data)

   # compute value update if relevant
   value_change = None
   if event in [StreamEventEnum.KEY_UPDATED, StreamEventEnum.KEY_COMPLETED]:
      prev_value = prev_partial.partial_obj.dict()[updated_key]
      curr_value = partial_obj.dict()[updated_key]
      if type(prev_value) == str and type(curr_value) == str:
         value_change = curr_value.replace(prev_value, "")
      else:
         value_change = curr_value
      print(event, updated_key, value_change)
   
   return StreamingObject[schema_model](
      value_change=value_change,
      updated_key=updated_key,
      event=event,
      partial_obj=partial_obj
   )

def parse_streamed_json(substring: str) -> tuple[dict, StreamEventEnum]:
    fixed_json_str, fix_reason = fix_truncated_json(substring)

    event = StreamEventEnum.KEY_UPDATED
    if fix_reason == JsonFixEnum.UNCLOSED_OBJECT:
        if fixed_json_str.count("\"") == 0:
            event = StreamEventEnum.OBJECT_CREATED
        else:
            event = StreamEventEnum.KEY_COMPLETED
    
    return json.loads(fixed_json_str), event