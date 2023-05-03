from json import loads as json_loads

import pytest

from gpt_json.truncation import fix_truncated_json, is_truncated


def test_is_truncated():
    assert is_truncated('{ "key": "value", ') == True
    assert is_truncated('[{"key": "value"}, ') == True
    assert is_truncated('{ "key": "value"}') == False
    assert is_truncated('[{"key": "value"}]') == False
    assert is_truncated('{"key": "value",') == True
    assert is_truncated('random_string') == False


@pytest.mark.parametrize(
    "broken_string,expected",
    [
        ('{ "key1": "value1", "key2": "value2", ', { "key1": "value1", "key2": "value2"}),
        ('[{"key1": "value1"}, {"key2": "value2"}, ', [{"key1": "value1"}, {"key2": "value2"}]),
        ('{ "key1": "value1", "key2": { "nestedKey": "nestedValue",', { "key1": "value1", "key2": { "nestedKey": "nestedValue"}}),
        ('[{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3", "key4": "value4",', [{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3", "key4": "value4"}]),
        ('[{"key1": "value1"}, {"key2": [1, 2, 3,', [{"key1": "value1"}, {"key2": [1, 2, 3]}]),
        ('{ "key1": "value1", "key2": "value2"}', { "key1": "value1", "key2": "value2"}),
        ('[{"key1": "value1"}, {"key2": "value2"}]', [{"key1": "value1"}, {"key2": "value2"}]),
        # test case with an escaped quote
        ('[{"key1": "value1"}, {"key2": "value2 \\" and a quote', [{"key1": "value1"}, {"key2": "value2 \" and a quote"}]),
        ('[{"key1": "value1"}, {"key2": "value2 start\n\n', [{"key1": "value1"}, {"key2": "value2 start"}]),
        # observed examples
        ('[{"key1": [123]', [{"key1": [123]}]),
        ('{"key1": [\n"abc",\n "def', {"key1": ["abc", "def"]})
    ]
)
def test_fix_truncated_json(broken_string, expected):
    print("BROKEN", broken_string)
    print("EXPECTED", expected)
    print("ACTUAL", fix_truncated_json(broken_string))
    assert json_loads(fix_truncated_json(broken_string)) == expected
