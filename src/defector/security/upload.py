"""Upload validation.

Three things are deliberately not trusted.

The client-supplied filename never reaches the filesystem. Writing it directly
is a path traversal vector, so the stored name is always a fresh uuid4 and the
extension comes from the format Pillow actually decoded.

The client-supplied content type is a header and proves nothing. The format is
established by decoding the image, not by reading Content-Type.

The request size is not bounded by FastAPI or uvicorn. Starlette's
max_part_size applies only to non-file parts, so file parts stream to a spooled
temporary file with no ceiling. The size is therefore checked here. In front of
a real deployment a reverse proxy limit belongs there as well, because by the
time this code runs the bytes have already been received.
"""

from dataclasses import dataclass
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from defector.ml.preprocess import open_image
from defector.settings import settings

# Pillow format name to the extension used on disk.
EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp"}


class UploadRejected(Exception):
    """Carries a message safe to return to the client."""


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    image: Image.Image
    stem: str
    filename: str
    image_format: str
    width: int
    height: int
    size_bytes: int


def validate(payload: bytes) -> ValidatedUpload:
    """Decode and check an upload, or raise UploadRejected."""
    if not payload:
        msg = "the uploaded file is empty"
        raise UploadRejected(msg)

    if len(payload) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / 1_048_576
        msg = f"file is larger than the {limit_mb:.0f} MB limit"
        raise UploadRejected(msg)

    # Decode once to establish the real format. Image.verify() would require
    # reopening the file afterwards, and the pixels are needed anyway for the
    # heatmap overlay.
    try:
        image = open_image(payload)
    except UnidentifiedImageError:
        msg = "the file is not a readable image"
        raise UploadRejected(msg) from None
    except Image.DecompressionBombError:
        msg = "the image resolution is unreasonably large"
        raise UploadRejected(msg) from None
    except OSError as error:
        msg = f"the image could not be decoded: {error}"
        raise UploadRejected(msg) from None

    image_format = (image.format or "").upper()
    if image_format not in settings.allowed_image_formats:
        allowed = ", ".join(settings.allowed_image_formats)
        msg = f"format {image_format or 'unknown'} is not supported; allowed: {allowed}"
        raise UploadRejected(msg)

    stem = uuid4().hex
    return ValidatedUpload(
        image=image,
        stem=stem,
        filename=f"{stem}{EXTENSIONS[image_format]}",
        image_format=image_format,
        width=image.width,
        height=image.height,
        size_bytes=len(payload),
    )
