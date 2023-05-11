import copy
from enum import Enum
from typing import Generic, Type, TypeVar

from pydantic import BaseModel, create_model

SchemaType = TypeVar("SchemaType", bound=BaseModel)

from typing import *


class OptionalConfig:
  arbitrary_types_allowed = True


class StreamEventEnum(Enum):
    OBJECT_CREATED = "OBJECT_CREATED"
    KEY_UPDATED = "KEY_UPDATED"
    KEY_COMPLETED = "KEY_COMPLETED"
    OTHER = "OTHER"


class StreamingObject(Generic[SchemaType]):
  schema_model: Type[SchemaType] = None

  updated_key: str | None
  value_change: str | None
  event: StreamEventEnum
  partial_obj: SchemaType

  def __init__(self, obj_data: dict, prev_partial: 'StreamingObject[SchemaType]', proposed_event: StreamEventEnum, value_change: str | None = None) -> None:
     super().__init__()
     
     raw_recent_key = list(obj_data.keys())[-1] if obj_data else None
     self.updated_key = raw_recent_key if raw_recent_key in self.schema_model.__fields__ else None
     
     self.event = proposed_event
     if proposed_event == StreamEventEnum.KEY_UPDATED and self.updated_key is None:
        self.event = StreamEventEnum.OTHER
     
     # TODO: this is hacky. ideally we want pydantic to implement Partial[SchemaType]
     # https://github.com/pydantic/pydantic/issues/1673
     cleaned_obj_data = {field: "" for field in self.schema_model.__fields__.keys()}
     cleaned_obj_data.update({k:v for k,v in obj_data.items() if v is not None})
     self.partial_obj = self.schema_model(**cleaned_obj_data)

     # compute value update if relevant
     if self.event in [StreamEventEnum.KEY_UPDATED, StreamEventEnum.KEY_COMPLETED]:
        prev_value = prev_partial.partial_obj.dict()[self.updated_key]
        self.value_change = self.partial_obj.dict()[self.updated_key].replace(prev_value, "")
     
  def __class_getitem__(cls, item):
    new_cls = super().__class_getitem__(item)
    new_cls.schema_model = item
    return new_cls
