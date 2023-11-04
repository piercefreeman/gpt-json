from json import loads as json_loads

import pytest

from gpt_json.transformations import JsonFixEnum, fix_bools, fix_truncated_json


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
    _, fix = fix_truncated_json(input_string)
    is_truncated = fix is not None
    assert is_truncated == expected


@pytest.mark.parametrize(
    "broken_string,expected,expected_fix_reason",
    [
        (
            '{ "key1": "value1", "key2": "value2", ',
            {"key1": "value1", "key2": "value2"},
            JsonFixEnum.UNCLOSED_OBJECT,
        ),
        (
            '[{"key1": "value1"}, {"key2": "value2"}, ',
            [{"key1": "value1"}, {"key2": "value2"}],
            JsonFixEnum.UNCLOSED_OBJECT,
        ),
        (
            '{ "key1": "value1", "key2": { "nestedKey": "nestedValue",',
            {"key1": "value1", "key2": {"nestedKey": "nestedValue"}},
            JsonFixEnum.UNCLOSED_OBJECT,
        ),
        (
            '[{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3", "key4": "value4",',
            [
                {"key1": "value1"},
                {"key2": "value2"},
                {"key3": "value3", "key4": "value4"},
            ],
            JsonFixEnum.UNCLOSED_OBJECT,
        ),
        (
            '[{"key1": "value1"}, {"key2": [1, 2, 3,',
            [{"key1": "value1"}, {"key2": [1, 2, 3]}],
            JsonFixEnum.UNCLOSED_OBJECT,
        ),
        (
            '{ "key1": "value1", "key2": "value2"}',
            {"key1": "value1", "key2": "value2"},
            None,
        ),
        (
            '[{"key1": "value1"}, {"key2": "value2"}]',
            [{"key1": "value1"}, {"key2": "value2"}],
            None,
        ),
        # test case with an escaped quote
        (
            '[{"key1": "value1"}, {"key2": "value2 \\" and a quote',
            [{"key1": "value1"}, {"key2": 'value2 " and a quote'}],
            JsonFixEnum.UNCLOSED_VALUE,
        ),
        (
            '[{"key1": "value1"}, {"key2": "value2 start\n\n',
            [{"key1": "value1"}, {"key2": "value2 start"}],
            JsonFixEnum.UNCLOSED_VALUE,
        ),
        # observed examples
        ('[{"key1": [123]', [{"key1": [123]}], JsonFixEnum.UNCLOSED_OBJECT),
        (
            '{"key1": [\n"abc",\n "def',
            {"key1": ["abc", "def"]},
            JsonFixEnum.UNCLOSED_VALUE,
        ),
        # streaming-specific examples
        ("[\n", [], JsonFixEnum.UNCLOSED_OBJECT),
        ("[\n", [], JsonFixEnum.UNCLOSED_OBJECT),
        ("{\n", {}, JsonFixEnum.UNCLOSED_OBJECT),
        ('[{"', [{"": None}], JsonFixEnum.UNCLOSED_KEY),
        ('[{"broken_key', [{"broken_key": None}], JsonFixEnum.UNCLOSED_KEY),
        (
            '[{"key_with_no_value":',
            [{"key_with_no_value": None}],
            JsonFixEnum.MISSING_VALUE,
        ),
        (
            '[{"another_key_with_no_value"',
            [{"another_key_with_no_value": None}],
            JsonFixEnum.MISSING_VALUE,
        ),
        (
            '{\n"student_model": "The student.",\n "tutor_response": "Good!"',
            {"student_model": "The student.", "tutor_response": "Good!"},
            JsonFixEnum.UNCLOSED_OBJECT,
        ),
        (
            '{\n"text": "Test", "numerical": 123, "reason": true, "sub_element": { "name": "Test" }, "items": ["Item 1", "Item 2 \n\n\n',
            {
                "text": "Test",
                "numerical": 123,
                "reason": True,
                "sub_element": {"name": "Test"},
                "items": ["Item 1", "Item 2"],
            },
            JsonFixEnum.UNCLOSED_VALUE,
        ),
        (
            '{"text": "Test"}}',
            {
                "text": "Test",
            },
            JsonFixEnum.DROP_TRAILING_JSON,
        ),
    ],
)
def test_fix_truncated_json(broken_string, expected, expected_fix_reason):
    fixed_string, fix_reason = fix_truncated_json(broken_string)

    print("BROKEN", broken_string)
    print("EXPECTED", expected)
    print("ACTUAL", fixed_string)

    assert json_loads(fixed_string) == expected

    print("EXPECTED FIX REASON", expected_fix_reason)
    print("ACTUAL FIX REASON", fix_reason)

    assert fix_reason == expected_fix_reason


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
