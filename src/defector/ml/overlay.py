"""Render the anomaly map as a heatmap over the original photograph.

Two things have to be right for the picture to mean what a user thinks it means.

Geometry: PatchCore returns the map at the pre-processor's size (224x224 here),
while uploads are full resolution, so the map is resized back before blending
and the highlighted region lines up with what was actually sent.

Opacity: the map is already normalised around the pixel threshold, where 0.5 is
the decision boundary. Blending it uniformly paints normal regions in mid-palette
colours and makes a defect-free part look suspicious. Instead the blend weight
is driven per pixel by how far the score sits above the threshold, so anything
below it stays untouched. A good part therefore renders as the original photo
with nothing highlighted, which is the correct message.

Overlays are written to disk rather than returned as base64: the history view
shows the same images again, so a file served by its own endpoint gets browser
caching and keeps the JSON payload a third smaller.
"""

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from anomalib.visualization import visualize_anomaly_map
from PIL import Image

from defector.settings import settings

# Peak opacity of the heatmap where the score is at its maximum.
MAX_ALPHA = 0.75


def render(
    anomaly_map: torch.Tensor,
    original: Image.Image,
    *,
    pixel_threshold: float = 0.5,
    filename: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Blend the map over the original and save a PNG. Returns the path."""
    blended = compose(anomaly_map, original, pixel_threshold=pixel_threshold)

    directory = output_dir or settings.heatmaps_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (filename or f"{uuid4().hex}.png")
    blended.save(path, format="PNG")
    return path


def compose(
    anomaly_map: torch.Tensor,
    original: Image.Image,
    *,
    pixel_threshold: float = 0.5,
) -> Image.Image:
    """Heatmap over the photograph, transparent below the threshold."""
    scores = _squeeze(anomaly_map)
    heatmap = (
        visualize_anomaly_map(scores, colormap=True, normalize=False)
        .convert("RGB")
        .resize(original.size, Image.BILINEAR)
    )

    # Per-pixel weight: zero at and below the threshold, MAX_ALPHA at 1.0.
    headroom = max(1.0 - pixel_threshold, 1e-6)
    weight = ((scores.numpy() - pixel_threshold) / headroom).clip(0.0, 1.0) * MAX_ALPHA
    alpha = Image.fromarray((weight * 255).astype(np.uint8), mode="L").resize(
        original.size, Image.BILINEAR
    )

    base = original.convert("RGB")
    return Image.composite(heatmap, base, alpha)


def render_map_only(
    anomaly_map: torch.Tensor,
    size: tuple[int, int] | None = None,
) -> Image.Image:
    """The colour-mapped anomaly map on its own, without the photograph."""
    heatmap = visualize_anomaly_map(_squeeze(anomaly_map), colormap=True, normalize=False)
    return heatmap.convert("RGB").resize(size, Image.BILINEAR) if size else heatmap.convert("RGB")


def _squeeze(anomaly_map: torch.Tensor) -> torch.Tensor:
    """visualize_anomaly_map expects (H, W); detections carry (1, H, W)."""
    return anomaly_map.squeeze(0) if anomaly_map.dim() == 3 else anomaly_map
