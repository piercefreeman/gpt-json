from gpt_json.models import JsonFixEnum


def build_stack(json_str):
    stack = []
    fixed_str = ""
    last_i = -1
    open_quotes = False

    # a flag indicating whether we've seen a comma or colon most recently
    # since last opening/closing a dict or list
    last_seen_comma_or_colon = None

    for i, char in enumerate(json_str):
        if not open_quotes:
            # opening a new nested
            if char in "{[":
                stack.append(char)
                last_seen_comma_or_colon = None
            # closing a nested
            elif char in "}]":
                if len(stack) == 0:
                    break
                stack.pop()
                last_seen_comma_or_colon = None
            if char in ",:":
                last_seen_comma_or_colon = char
        # opening or closing a string, only it's not escaped
        if char == '"' and i > 0 and json_str[i - 1] != "\\":
            open_quotes = not open_quotes

        fixed_str += char
        last_i = i + 1

    unparsed_str = json_str[last_i:]
    return (stack, fixed_str, open_quotes, last_seen_comma_or_colon, unparsed_str)


def _is_missing_dict_value(stack, fixed_str, open_quotes, last_seen_comma_or_colon):
    # check if we're missing a dict value in the json string
    inside_dict = len(stack) > 0 and stack[-1] == "{"
    inside_dict_key = inside_dict and open_quotes and last_seen_comma_or_colon != ":"
    just_before_dict_value = (
        inside_dict and not open_quotes and last_seen_comma_or_colon == ":"
    )
    just_closed_dict_key = (
        inside_dict and not open_quotes and fixed_str.strip()[-1] == '"'
    )
    just_closed_dict_value = (
        inside_dict
        and not open_quotes
        and fixed_str.strip()[-1] == '"'
        and last_seen_comma_or_colon == ":"
    )
    missing_dict_value = (
        inside_dict_key or just_before_dict_value or just_closed_dict_key
    ) and not just_closed_dict_value
    return missing_dict_value


def is_truncated(json_str):
    """
    Check if the json string is truncated by checking if the number of opening
    brackets is greater than the number of closing brackets.

    """
    stack, _, _, _, _ = build_stack(json_str)
    return len(stack) > 0


def fix_truncated_json(json_str) -> tuple[str, JsonFixEnum | None]:
    """
    Simple json parser that attempts to fix truncated json that might
    be caused by response streaming or the API response being too long.

    Returns a tuple of (fixed_json_string, fix_type)
    """
    stack, fixed_str, open_quotes, last_seen_colon_or_comma, unparsed_str = build_stack(
        json_str
    )
    missing_value = _is_missing_dict_value(
        stack, fixed_str, open_quotes, last_seen_colon_or_comma
    )
    is_truncated = len(stack) > 0
    if not is_truncated:
        if not unparsed_str.strip():
            return json_str, None
        else:
            return fixed_str, JsonFixEnum.DROP_TRAILING_JSON

    fixed_str = fixed_str.strip()

    # propose null cases to handle missing values in truncated JSON string
    if open_quotes:
        fixed_str += '"'
    if missing_value:
        fixed_str = fixed_str.rstrip(":") + ": null"

    # Ensure we don't have trailing commas
    fixed_str = fixed_str.strip().rstrip(",")

    # If we still have nested items remaining in our stack,
    # unwind it into the fixed string
    if stack:
        # Unwind the stack by filling it with the closing character
        # of the current nested level
        close_stack = ["]" if char == "[" else "}" for char in stack]
        fixed_str += "".join(close_stack[::-1])

    # if the fixed string is valid JSON, return it
    fix = JsonFixEnum.UNCLOSED_OBJECT
    if open_quotes:
        fix = JsonFixEnum.UNCLOSED_KEY if missing_value else JsonFixEnum.UNCLOSED_VALUE
    elif missing_value:
        fix = JsonFixEnum.MISSING_VALUE

    return fixed_str, fix


def fix_bools(json_str):
    """
    The model will relatively commonly return booleans as capitalized values because of the
    usage of caps in other languages common in the training set (like Python).

    """
    modified = False
    open_quotes = False
    fixed_str = ""

    i = 0
    while i < len(json_str):
        char = json_str[i]

        # Check if the current character is an opening or closing quote
        if char == '"' and i > 0 and json_str[i - 1] != "\\":
            open_quotes = not open_quotes

        # If not inside a string, check for "True" or "False" to replace
        if not open_quotes:
            if json_str[i : i + 4] == "True":
                fixed_str += "true"
                modified = True
                i += 3  # Skip the remaining characters of "True"
            elif json_str[i : i + 5] == "False":
                fixed_str += "false"
                modified = True
                i += 4  # Skip the remaining characters of "False"
            else:
                fixed_str += char
        else:
            fixed_str += char
        i += 1

    return fixed_str, modified
