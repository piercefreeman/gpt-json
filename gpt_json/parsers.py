import json
from re import DOTALL, finditer

from gpt_json.models import ResponseType
from gpt_json.transformations import is_truncated
from gpt_json.types_streaming import StreamEventEnum


def find_json_response(full_response, extract_type):
    """
    Takes a full response that might contain other strings and attempts to extract the JSON payload.
    Has support for truncated JSON where the JSON begins but the token window ends before the json is
    is properly closed.

    """
    # Deal with fully included responses as well as truncated responses that only have one
    if extract_type == ResponseType.LIST:
        extracted_responses = list(finditer(r"(\[[^\]]*$|\[.*\])", full_response, flags=DOTALL))
    elif extract_type == ResponseType.DICTIONARY:
        extracted_responses = list(finditer(r"({[^}]*$|{.*})", full_response, flags=DOTALL))
    else:
        raise ValueError("Unknown extract_type")

    if not extracted_responses:
        print(f"Unable to find any responses of the matching type `{extract_type}`: `{full_response}`")
        return None

    if len(extracted_responses) > 1:
        print("Unexpected response > 1, continuing anyway...", extracted_responses)

    extracted_response = extracted_responses[0]

    if is_truncated(extracted_response.group(0)):
        # Start at the same location and just expand to the end of the message
        extracted_response = full_response[extracted_response.start():]
    else:
        extracted_response = extracted_response.group(0)

    return extracted_response


def _is_valid_json(substring: str) -> bool:
    try:
        json.loads(substring)
        return True
    except json.decoder.JSONDecodeError:
        return False

def _fix_broken_json(substring: str) -> tuple[str, str]:
    """Only works for a single JSON object with no nested objects and whose keys and values are strings.
    
    Returns (fixed_json_string, complete_reason)
    
    complete_reason is one of: ["no_fix_needed", "unclosed_object", "unclosed_value", "missing_value", "unclosed_key"]
    """
    if not len(substring.strip()):
        return "{}", "empty_string"
    
    # If the substring is already valid JSON, return it as is
    if _is_valid_json(substring):
        return substring, "no_fix_needed"
    
    # fix an unclosed object
    fixed_substring = substring + "}"
    if _is_valid_json(fixed_substring):
        return fixed_substring, "unclosed_object"

    # fix an unclosed object with a dangling comma
    fixed_substring = substring.strip()[:-1] + "}"
    if _is_valid_json(fixed_substring):
        return fixed_substring, "unclosed_object"

    # fix an unclosed value
    fixed_substring = substring + '"}'
    if _is_valid_json(fixed_substring):
        return fixed_substring, "unclosed_value"

    # fix a missing value
    fixed_substring = substring + ' null}'
    if _is_valid_json(fixed_substring):
        return fixed_substring, "missing_value"

    # fix a missing value w/ colon
    fixed_substring = substring + ': null}'
    if _is_valid_json(fixed_substring):
        return fixed_substring, "missing_value"

    # fix an unclosed key
    fixed_substring = substring + '": null}'
    if _is_valid_json(fixed_substring):
        return fixed_substring, "unclosed_key"

    # fix an unclosed key that has only the starting quote
    fixed_substring = substring[:-1] + '}'
    if _is_valid_json(fixed_substring):
        return fixed_substring, "unclosed_key"
    
    return fixed_substring, None

def parse_streamed_json(substring: str) -> tuple[dict, StreamEventEnum]:
    fixed_json_str, fix_reason = _fix_broken_json(substring)

    event = StreamEventEnum.KEY_UPDATED
    if fix_reason == "unclosed_object":
        if fixed_json_str.count("\"") == 0:
            event = StreamEventEnum.OBJECT_CREATED
        else:
            event = StreamEventEnum.KEY_COMPLETED

    
    return json.loads(fixed_json_str), event
    