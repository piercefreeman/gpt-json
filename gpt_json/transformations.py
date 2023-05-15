import json
from enum import Enum


def build_stack(json_str):
    stack = []
    fixed_str = ''
    open_quotes = False

    for i, char in enumerate(json_str):
        if not open_quotes:
            # opening a new nested
            if char in "{[":
                stack.append(char)
            # closing a nested
            elif char in "}]":
                stack.pop()
        # opening or closing a string, only it's not escaped
        if char == '"' and i > 0 and json_str[i-1] != "\\":
            open_quotes = not open_quotes

        fixed_str += char
    
    return (stack, fixed_str, open_quotes)

def is_truncated(json_str):
    """
    Check if the json string is truncated by checking if the number of opening
    brackets is greater than the number of closing brackets.

    """
    stack, _, _ = build_stack(json_str)
    return len(stack) > 0

class JsonFixEnum(Enum):
    UNCLOSED_OBJECT = "unclosed_object"
    UNCLOSED_KEY = "unclosed_key"
    UNCLOSED_VALUE = "unclosed_value"
    MISSING_VALUE = "missing_value"

def fix_truncated_json(json_str) -> tuple[str, JsonFixEnum | None]:
    """
    Simple json parser that attempts to fix truncated json that might
    be caused by response streaming or the API response being too long.

    Returns a tuple of (fixed_json_string, fix_type)
    """
    stack, fixed_str, open_quotes = build_stack(json_str)
    is_truncated = len(stack) > 0
    if not is_truncated:
        return json_str, None

    fixed_str = fixed_str.strip()

    # propose null cases to handle missing values in truncated JSON string
    proposed_fixed_strs = [fixed_str, fixed_str + ' null']
    if open_quotes:
        proposed_fixed_strs = [fixed_str + '"', fixed_str + '": null']

    for idx, fixed_str in enumerate(proposed_fixed_strs):
        is_null_case = idx == 1
        
        # Ensure we don't have trailing commas
        fixed_str = fixed_str.strip().rstrip(",")

        # If we still have nested items remaining in our stack,
        # unwind it into the fixed string
        if stack:
            # Unwind the stack by filling it with the closing character
            # of the current nested level
            close_stack = ["]" if char == "[" else "}" for char in stack]
            fixed_str += ''.join(close_stack[::-1])
        
        # if the fixed string is valid JSON, return it
        fix = JsonFixEnum.UNCLOSED_OBJECT
        if open_quotes:
            fix = JsonFixEnum.UNCLOSED_KEY if is_null_case else JsonFixEnum.UNCLOSED_VALUE
        elif is_null_case:
            fix = JsonFixEnum.MISSING_VALUE
        try:
            json.loads(fixed_str)
            return fixed_str, fix
        except json.decoder.JSONDecodeError:
            pass

    raise ValueError("Unable to fix truncated JSON string")


def fix_bools(json_str):
    """
    The model will relatively commonly return booleans as capitalized values because of the
    usage of caps in other languages common in the training set (like Python).

    """
    modified = False
    open_quotes = False
    fixed_str = ''

    i = 0
    while i < len(json_str):
        char = json_str[i]

        # Check if the current character is an opening or closing quote
        if char == '"' and i > 0 and json_str[i-1] != "\\":
            open_quotes = not open_quotes

        # If not inside a string, check for "True" or "False" to replace
        if not open_quotes:
            if json_str[i:i+4] == "True":
                fixed_str += "true"
                modified = True
                i += 3  # Skip the remaining characters of "True"
            elif json_str[i:i+5] == "False":
                fixed_str += "false"
                modified = True
                i += 4  # Skip the remaining characters of "False"
            else:
                fixed_str += char
        else:
            fixed_str += char
        i += 1

    return fixed_str, modified
