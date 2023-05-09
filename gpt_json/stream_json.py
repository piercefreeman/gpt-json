import json


def stream_json(oai_generator):
    """Streams JSON objects from an OpenAI Completion generator
    TODO: write better documentation
    """
    # TODO: support other mode configurations 
    full_keys_only=True
    full_values_only=False
    
    next(oai_generator) # the first token just mentions the assistant role

    def should_yield(prev, curr, fix, prev_fix):
        if prev is None:
            return True
        
        # never yield if the json data is identical
        if list(prev.keys()) == list(curr.keys()) and list(prev.values()) == list(curr.values()):
            # UNLESS previous fix was unclosed value and we are in full keys only mode
            if prev_fix == "unclosed_value" and full_keys_only:
                return True, "completed_value"
            # OR previous fix was unclosed key and we are in full values only mode
            if prev_fix == "unclosed_key" and full_keys_only:
                return True, "completed_key"
            return False

        # if full keys only, we shouldn't yield if we're just closing a key
        if full_keys_only and fix == "unclosed_key":
            return False
        
        
        # if full values only, we shouldn't yield if we're just closing a value
        if full_values_only and fix == "unclosed_value":
            return False

        return True

    prev_json_data = None
    prev_yield_data = None
    prev_fix = None
    partial_json_str = ""
    for token in oai_generator:
        if token["choices"][0]["finish_reason"] == "stop":
            break

        curr_token = token["choices"][0]["delta"]["content"]
        partial_json_str += curr_token
        recovered_json_str, fix_reason = complete_json(partial_json_str) 

        json_data = json.loads(recovered_json_str)
        if should_yield(prev_json_data, json_data, fix_reason, prev_fix):
            if prev_yield_data is None:
                event = "create_object"
            elif len(json_data.keys()) > len(prev_yield_data.keys()):
                event = "new_key"
            elif json_data == prev_yield_data:
                event = "completed_value"
            else:
                event = "value_changed"
            
            if not (event == "value_changed" and curr_token.strip() == "\""):
                if event == "create_object":
                    curr_token = None

                if event == "completed_value":
                    curr_token = json_data[list(json_data.keys())[-1]]

                if event == "new_key":
                    curr_token = list(json_data.keys())[-1]
                
                # TODO: handle these corner cases better
                # it's due to OpenAI's tokenization
                if curr_token and curr_token.strip() in [".\",", "?\"", ".\"", "?\",", "!\"", "!\",", "!\"", "\","]:
                    curr_token = curr_token.replace("\"", "").replace(",", "")

                yield json_data, curr_token, event
                prev_yield_data = json_data

        prev_json_data = json_data
        prev_fix = fix_reason


def complete_json(substring):
    """Only works for a single JSON object with no nested objects and whose keys and values are strings.
    
    Returns (fixed_json_string, complete_reason)
    
    complete_reason is one of: ["no_fix_needed", "unclosed_object", "unclosed_value", "missing_value", "unclosed_key"]
    """
    
    if not len(substring.strip()):
        return "{}", "empty_string"
    
    # If the substring is already valid JSON, return it as is
    fixed_substring = substring
    try:
        json.loads(fixed_substring)
        return fixed_substring, "no_fix_needed"
    except json.decoder.JSONDecodeError as ex:
        pass

    # fix an unclosed object
    fixed_substring = substring + "}"
    try:
        json.loads(fixed_substring)
        return fixed_substring, "unclosed_object"
    except json.decoder.JSONDecodeError as ex:
        pass

    # fix an unclosed object with a dangling comma
    fixed_substring = substring.strip()[:-1] + "}"
    try:
        json.loads(fixed_substring)
        return fixed_substring, "unclosed_object"
    except json.decoder.JSONDecodeError as ex:
        pass

    # fix an unclosed value
    fixed_substring = substring + '"}'
    try:
        json.loads(fixed_substring)
        return fixed_substring, "unclosed_value"
    except json.decoder.JSONDecodeError as ex:
        pass

    # fix a missing value
    fixed_substring = substring + ' null}'
    try:
        json.loads(fixed_substring)
        return fixed_substring, "missing_value"
    except json.decoder.JSONDecodeError as ex:
        pass

    # fix a missing value w/ colon
    fixed_substring = substring + ': null}'
    try:
        json.loads(fixed_substring)
        return fixed_substring, "missing_value"
    except json.decoder.JSONDecodeError as ex:
        pass

    # fix an unclosed key
    fixed_substring = substring + '": null}'
    try:
        json.loads(fixed_substring)
        return fixed_substring, "unclosed_key"
    except json.decoder.JSONDecodeError as ex:
        pass

    # fix an unclosed key that has only the starting quote
    fixed_substring = substring[:-1] + '}'
    try:
        json.loads(fixed_substring)
        return fixed_substring, "unclosed_key"
    except json.decoder.JSONDecodeError as ex:
        pass
    
    return fixed_substring, None
        

test_json_str = """{
    "student_model": "The student",
    "actionID": "action goes here"
}"""
if __name__ == "__main__":
    def oai_generator_dummy():
        yield {} # dummy assistant role response
        for i in range(len(test_json_str)):
            yield {
                "choices": [
                    {
                        "delta": {
                            "content": test_json_str[i]
                        },
                        "finish_reason": "null",
                    }
                ]
            }
    
    oai_generator = oai_generator_dummy()
    for json_data, curr_token, event in stream_json(oai_generator):
        print(json_data, curr_token, event)

