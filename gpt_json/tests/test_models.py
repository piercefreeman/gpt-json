import pytest
import gpt_json.models as models_file   
from json import loads as json_loads, dumps as json_dumps
from enum import Enum

@pytest.mark.parametrize("model_file", [models_file])
def test_serializable_enums(model_file):
    """
    All our enums should be serializable as JSON

    """
    found_enums = 0
    for obj in model_file.__dict__.values():
        if isinstance(obj, type) and issubclass(obj, Enum):
            found_enums += 1
            for enum_value in obj:
                assert enum_value.value == json_loads(json_dumps(enum_value))

    # Every file listed in pytest should have at least one enum
    assert found_enums > 0
