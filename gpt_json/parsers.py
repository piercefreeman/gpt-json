from re import DOTALL, finditer

from gpt_json.models import ResponseType
from gpt_json.transformations import is_truncated


def find_json_response(full_response, extract_type):
    """
    Takes a full response that might contain other strings and attempts to extract the JSON payload.
    Has support for truncated JSON where the JSON begins but the token window ends before the json is
    is properly closed.

    """
    # Deal with fully included responses as well as truncated responses that only have one
    if extract_type == ResponseType.DICTIONARY:
        extracted_responses = list(
            finditer(r"({[^}]*$|{.*})", full_response, flags=DOTALL)
        )
    else:
        raise ValueError("Unknown extract_type")

    if not extracted_responses:
        print(
            f"Unable to find any responses of the matching type `{extract_type}`: `{full_response}`"
        )
        return None

    if len(extracted_responses) > 1:
        print("Unexpected response > 1, continuing anyway...", extracted_responses)

    extracted_response = extracted_responses[0]

    if is_truncated(extracted_response.group(0)):
        # Start at the same location and just expand to the end of the message
        extracted_str = full_response[extracted_response.start() :]
    else:
        extracted_str = extracted_response.group(0)

    return extracted_str
