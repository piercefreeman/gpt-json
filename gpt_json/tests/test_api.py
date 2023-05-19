from gpt_json import GPTJSON, GPTMessage, GPTMessageRole, GPTModelVersion, ResponseType


def test_exports_variables():
    """
    Test that the library exports the correct models for end users
    """
    assert GPTJSON is not None
    assert GPTMessage is not None
    assert GPTMessageRole is not None
    assert GPTModelVersion is not None
    assert ResponseType is not None
