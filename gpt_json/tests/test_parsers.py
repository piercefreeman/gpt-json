import pytest

from gpt_json.models import ResponseType
from gpt_json.parsers import find_json_response


@pytest.mark.parametrize(
    "input_string,expected,extract_type",
    [
        ('This message is truncated: [{"key1": [123]', '[{"key1": [123]', ResponseType.LIST),
        ('This message is truncated: [{"key1": [123', '[{"key1": [123', ResponseType.LIST),
        ('This message is truncated: [{"key1": "abc"', '[{"key1": "abc"', ResponseType.LIST),
        ('This message is truncated: {"key": "value", "list": [1, 2, 3', '{"key": "value", "list": [1, 2, 3', ResponseType.DICTIONARY),
    ]
)
def test_find_json_response(input_string, expected, extract_type):
    assert find_json_response(input_string, extract_type) == expected
