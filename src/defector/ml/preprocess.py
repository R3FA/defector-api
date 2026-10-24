"""Image loading and the transforms shared by the detectors and the classifier.

Two different preprocessing paths exist and they must not be confused:

- PatchCore uses the pre-processor stored inside its own checkpoint, so the
  geometry at inference is exactly the geometry used during training. It is
  reached through ``model.pre_processor`` and is not rebuilt here.
- The category classifier uses the transforms that ship with the ResNet18
  ImageNet weights, because that is what its features were extracted with.
"""

import io
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

from defector.settings import settings

# Guard against decompression bomb uploads. Pillow warns at this value and
# raises at twice it.
Image.MAX_IMAGE_PIXELS = settings.max_image_pixels


@lru_cache(maxsize=1)
def classifier_transform():
    """Preprocessing that matches the ResNet18 ImageNet weights."""
    return ResNet18_Weights.IMAGENET1K_V1.transforms()


def open_image(source: bytes | str | Path) -> Image.Image:
    """Decode an upload or a file into an RGB image.

    convert() returns a new object that does not carry the format attribute,
    which Pillow only sets on images opened from a source. The format is
    reattached because upload validation establishes the real file type from it
    rather than trusting the client's Content-Type.
    """
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(handle) as decoded:
        image_format = decoded.format
        converted = decoded.convert("RGB")
    converted.format = image_format
    return converted


def to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB image to (1, 3, H, W) float32 in [0, 1].

    Same convention as anomalib's read_image, which is what the models were
    trained and evaluated against.
    """
    array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = array.view(image.height, image.width, 3).permute(2, 0, 1)
    return tensor.to(torch.float32).div_(255.0).unsqueeze(0)


def load_tensor(source: bytes | str | Path) -> tuple[torch.Tensor, Image.Image]:
    """Convenience for the scripts: returns the tensor and the decoded image.

    The image is returned as well because the heatmap overlay needs the original
    pixels, and decoding twice would be wasteful.
    """
    image = open_image(source)
    return to_tensor(image), image
