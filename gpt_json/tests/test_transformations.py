from json import loads as json_loads

import pytest

from gpt_json.transformations import fix_bools, fix_truncated_json


@pytest.mark.parametrize(
    "input_string,expected",
    [
        ('{ "key": "value", ', True),
        ('[{"key": "value"}, ', True),
        ('{ "key": "value"}', False),
        ('[{"key": "value"}]', False),
        ('{"key": "value",', True),
        ("random_string", False),
    ],
)
def test_is_truncated(input_string: str, expected: bool):
    _, is_truncated = fix_truncated_json(input_string)
    assert is_truncated == expected


@pytest.mark.parametrize(
    "broken_string,expected",
    [
        (
            '{ "key1": "value1", "key2": "value2", ',
            {"key1": "value1", "key2": "value2"},
        ),
        (
            '[{"key1": "value1"}, {"key2": "value2"}, ',
            [{"key1": "value1"}, {"key2": "value2"}],
        ),
        (
            '{ "key1": "value1", "key2": { "nestedKey": "nestedValue",',
            {"key1": "value1", "key2": {"nestedKey": "nestedValue"}},
        ),
        (
            '[{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3", "key4": "value4",',
            [
                {"key1": "value1"},
                {"key2": "value2"},
                {"key3": "value3", "key4": "value4"},
            ],
        ),
        (
            '[{"key1": "value1"}, {"key2": [1, 2, 3,',
            [{"key1": "value1"}, {"key2": [1, 2, 3]}],
        ),
        ('{ "key1": "value1", "key2": "value2"}', {"key1": "value1", "key2": "value2"}),
        (
            '[{"key1": "value1"}, {"key2": "value2"}]',
            [{"key1": "value1"}, {"key2": "value2"}],
        ),
        # test case with an escaped quote
        (
            '[{"key1": "value1"}, {"key2": "value2 \\" and a quote',
            [{"key1": "value1"}, {"key2": 'value2 " and a quote'}],
        ),
        (
            '[{"key1": "value1"}, {"key2": "value2 start\n\n',
            [{"key1": "value1"}, {"key2": "value2 start"}],
        ),
        # observed examples
        ('[{"key1": [123]', [{"key1": [123]}]),
        ('{"key1": [\n"abc",\n "def', {"key1": ["abc", "def"]}),
    ],
)
def test_fix_truncated_json(broken_string, expected):
    fixed_string, _ = fix_truncated_json(broken_string)

    print("BROKEN", broken_string)
    print("EXPECTED", expected)
    print("ACTUAL", fixed_string)

    assert json_loads(fixed_string) == expected


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        # No change required
        (
            '{"status": "success", "data": {"isAdmin": true, "isActive": false}}',
            (
                '{"status": "success", "data": {"isAdmin": true, "isActive": false}}',
                False,
            ),
        ),
        # Fix "True" to "true"
        (
            '{"status": "success", "data": {"isAdmin": True, "isActive": false}}',
            (
                '{"status": "success", "data": {"isAdmin": true, "isActive": false}}',
                True,
            ),
        ),
        # Fix "False" to "false"
        (
            '{"status": "success", "data": {"isAdmin": true, "isActive": False}}',
            (
                '{"status": "success", "data": {"isAdmin": true, "isActive": false}}',
                True,
            ),
        ),
        # Fix both "True" and "False"
        (
            '{"status": "success", "data": {"isAdmin": True, "isActive": False}}',
            (
                '{"status": "success", "data": {"isAdmin": true, "isActive": false}}',
                True,
            ),
        ),
        # Test with string containing "True" and "False" as substrings
        (
            '{"status": "TrueValue", "data": {"isAdmin": True, "isActive": False, "attribute": "FalsePositive"}}',
            (
                '{"status": "TrueValue", "data": {"isAdmin": true, "isActive": false, "attribute": "FalsePositive"}}',
                True,
            ),
        ),
    ],
)
def test_fix_bools(input_str, expected_output):
    assert fix_bools(input_str) == expected_output
