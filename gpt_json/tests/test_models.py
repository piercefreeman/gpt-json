import pytest
from pydantic import HttpUrl

from gpt_json.models import GPTMessage, GPTMessageRole, ImageContent


def test_gpt_message_validates_function():
    with pytest.raises(ValueError):
        GPTMessage(
            role=GPTMessageRole.SYSTEM,
            name="function_name",
            content="function_content",
        )

    with pytest.raises(ValueError):
        GPTMessage(
            role=GPTMessageRole.FUNCTION,
            content="function_content",
        )

    GPTMessage(
        role=GPTMessageRole.FUNCTION,
        name="function_name",
        content="function_content",
    )


def test_image_content_from_url():
    assert ImageContent.from_url("https://example.com/image.jpg") == ImageContent(
        image_url=ImageContent.ImageUrl(url=HttpUrl("https://example.com/image.jpg"))
    )


def test_image_content_from_bytes():
    assert ImageContent.from_bytes(b"image_bytes", "image/png") == ImageContent(
        image_url=ImageContent.ImageBytes(url="data:image/png;base64,aW1hZ2VfYnl0ZXM=")
    )
