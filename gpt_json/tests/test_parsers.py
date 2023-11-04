import pytest

from gpt_json.models import ResponseType
from gpt_json.parsers import find_json_response


@pytest.mark.parametrize(
    "input_string,expected,extract_type",
    [
        (
            'This message is truncated: {"items":[{"key1": [123]',
            '{"items":[{"key1": [123]',
            ResponseType.DICTIONARY,
        ),
        (
            'This message is truncated: {"items":[{"key1": [123',
            '{"items":[{"key1": [123',
            ResponseType.DICTIONARY,
        ),
        (
            'This message is truncated: {"items":[{"key1": "abc"',
            '{"items":[{"key1": "abc"',
            ResponseType.DICTIONARY,
        ),
        (
            'This message is truncated: {"key": "value", "list": [1, 2, 3',
            '{"key": "value", "list": [1, 2, 3',
            ResponseType.DICTIONARY,
        ),
        (
            'This message is truncated: {"text": "Test", "numerical": 123, "reason": true, "sub_element": { "name": "Test" }, "items": ["Item 1", "Item 2',
            '{"text": "Test", "numerical": 123, "reason": true, "sub_element": { "name": "Test" }, "items": ["Item 1", "Item 2',
            ResponseType.DICTIONARY,
        ),
        (
            'This message has an additional closing bracket: {"text": "Test"}}',
            '{"text": "Test"}}',
            ResponseType.DICTIONARY,
        ),
    ],
)
def test_find_json_response(input_string, expected, extract_type):
    assert find_json_response(input_string, extract_type) == expected
