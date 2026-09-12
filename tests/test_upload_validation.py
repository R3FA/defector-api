"""Upload validation: what is accepted, what is refused, and how it is named."""

import io

import pytest
from PIL import Image

from defector.security.upload import UploadRejected, validate
from defector.settings import settings


def _encode(image: Image.Image, image_format: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_accepts_png_and_reads_the_format_from_the_bytes(png_bytes: bytes) -> None:
    result = validate(png_bytes)
    assert result.image_format == "PNG"
    assert result.filename.endswith(".png")
    assert result.width == 64


def test_accepts_jpeg(jpeg_bytes: bytes) -> None:
    assert validate(jpeg_bytes).image_format == "JPEG"


def test_format_comes_from_content_not_from_a_claimed_extension() -> None:
    """A JPEG is stored as .jpg even if a caller would have named it .png."""
    payload = _encode(Image.new("RGB", (32, 32), "blue"), "JPEG")
    assert validate(payload).filename.endswith(".jpg")


def test_stored_name_is_a_uuid_so_a_client_filename_cannot_traverse() -> None:
    result = validate(_encode(Image.new("RGB", (8, 8)), "PNG"))
    assert len(result.stem) == 32
    assert all(c in "0123456789abcdef" for c in result.stem)
    assert "/" not in result.filename and ".." not in result.filename


def test_two_uploads_of_identical_bytes_get_distinct_names(png_bytes: bytes) -> None:
    assert validate(png_bytes).filename != validate(png_bytes).filename


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        (b"", "empty"),
        (b"#!/bin/sh\nrm -rf /\n", "not a readable image"),
    ],
)
def test_rejects_non_images(payload: bytes, fragment: str) -> None:
    with pytest.raises(UploadRejected, match=fragment):
        validate(payload)


def test_rejects_a_disallowed_format() -> None:
    payload = _encode(Image.new("RGB", (8, 8), "red"), "GIF")
    with pytest.raises(UploadRejected, match="not supported"):
        validate(payload)


def test_rejects_a_payload_over_the_limit() -> None:
    oversized = b"\x89PNG\r\n\x1a\n" + b"x" * settings.max_upload_bytes
    with pytest.raises(UploadRejected, match="larger than"):
        validate(oversized)
