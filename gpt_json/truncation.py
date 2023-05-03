
def is_truncated(json_str):
    stack = []
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

    return len(stack) > 0


def fix_truncated_json(json_str):
    """
    Simple json parser that attempts to fix truncated json that might
    be caused by the API response being too long.

    """
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

    fixed_str = fixed_str.strip()

    if open_quotes:
        fixed_str += '"'

    # Ensure we don't have trailing commas
    fixed_str = fixed_str.strip().rstrip(",")

    # If we still have nested items remaining in our stack,
    # unwind it into the fixed string
    if stack:
        # Unwind the stack by filling it with the closing character
        # of the current nested level
        close_stack = ["]" if char == "[" else "}" for char in stack]
        fixed_str += ''.join(close_stack[::-1])

    return fixed_str
