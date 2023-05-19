import pytest

from gpt_json.gpt import logger


def test_incorrect_logging_calls():
    """
    The `set_logger_level` fixture is injected automatically by pytest.
    """
    with pytest.raises(TypeError):
        logger.setLevel("test string", incorrect_arg="test string")  # type: ignore
