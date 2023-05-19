import logging

import pytest

from gpt_json.gpt import logger


@pytest.fixture(autouse=True)
def set_logger_level():
    """
    Fixture is useful both to get additional context on test failures, and to
    debug errors while calling the logger. Otherwise calls are no-op, even with incorrect
    parameters.

    """
    logger.setLevel(logging.DEBUG)
