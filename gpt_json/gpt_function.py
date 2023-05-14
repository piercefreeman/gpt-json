
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from gpt_json.gpt import GPTJSON
from gpt_json.models import GPTMessage

InputSchemaType = TypeVar("InputSchemaType", bound=BaseModel)
OutputSchemaType = TypeVar("OutputSchemaType", bound=BaseModel)

# TODO: generically type this guy
# class GPTFunction(Generic[InputSchemaType, OutputSchemaType]):
class GPTFunction():
    input_schema: InputSchemaType = None
    output_schema: OutputSchemaType = None

    def __init__(self, gpt_json: GPTJSON, msgs: list[GPTMessage], transforms: dict[str, Any]={}):
        self.gpt_json = gpt_json
        self.msgs = msgs
        self.transforms = transforms
    
    async def __call__(self, **format_variables: dict):
        for key, transform in self.transforms.items():
            if key in format_variables:
                format_variables[key] = transform(format_variables[key])
        
        output, _ = await self.gpt_json.run(self.msgs, format_variables=format_variables)
        return output

    # TODO: infer input schema type based on {vars} in prompt messages?
    # def __class_getitem__(cls, output_schema):
    #     new_cls = super().__class_getitem__(input_schema, output_schema)
    #     new_cls.output_schema = output_schema
    #     return new_cls
