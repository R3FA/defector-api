"""Anomaly detection for a single image.

Wraps one trained PatchCore model per category and returns the verdict, the
score, the threshold it was compared against, and the pixel-level anomaly map.

The threshold deserves care. Anomalib normalises pred_score against the
adaptive threshold it computed during validation, using
``((raw - threshold) / (max - min)) + 0.5`` clamped to [0, 1]. Comparing that
normalised score against ``post_processor.image_threshold``, which is on the
raw scale, silently produces wrong verdicts. The correct comparison is against
``normalized_image_threshold`` (0.5 by default), and pred_label already encodes
exactly that, so it is what gets used here.

Strictness is adjusted through ``image_sensitivity``: the effective boundary is
``1 - image_sensitivity``, so raising sensitivity lowers the boundary and flags
more parts as defective. In quality control a missed defect costs more than a
good part rejected, which is the argument for tuning it upward.
"""

import time
from dataclasses import dataclass

import torch
from anomalib.models import Patchcore

from defector.settings import settings


@dataclass(frozen=True, slots=True)
class Detection:
    category: str
    is_defective: bool
    score: float
    threshold: float
    defect_area_px: int
    duration_ms: int
    anomaly_map: torch.Tensor
    pred_mask: torch.Tensor


class Detector:
    """One trained PatchCore model, ready for single-image inference."""

    def __init__(self, category: str, model: Patchcore) -> None:
        self.category = category
        self._model = model

    @classmethod
    def load(cls, category: str) -> "Detector":
        checkpoint = settings.checkpoint_path(category)
        if not checkpoint.is_file():
            msg = f"no checkpoint for {category} at {checkpoint}; run scripts/02_train_detectors.py"
            raise FileNotFoundError(msg)
        # weights_only=False is required: the checkpoint carries pickled
        # pre-processor, post-processor and evaluator objects, and torch>=2.6
        # defaults to weights_only=True, which refuses them.
        model = Patchcore.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)
        return cls(category, model.eval())

    @property
    def backbone(self) -> str:
        """Backbone the checkpoint was trained with, not the one configured now."""
        return str(self._model.hparams["backbone"])

    @property
    def coreset_sampling_ratio(self) -> float:
        return float(self._model.hparams["coreset_sampling_ratio"])

    @property
    def threshold(self) -> float:
        """Decision boundary on the normalised scale, not the raw one."""
        return float(self._model.post_processor.normalized_image_threshold)

    @property
    def pixel_threshold(self) -> float:
        """Decision boundary for the anomaly map, on the normalised scale."""
        return float(self._model.post_processor.normalized_pixel_threshold)

    @property
    def sensitivity(self) -> float:
        return float(self._model.post_processor.image_sensitivity)

    def set_sensitivity(self, value: float) -> None:
        """Higher means stricter: the effective boundary becomes 1 - value."""
        if not 0.0 < value < 1.0:
            msg = f"sensitivity must be in (0, 1), got {value}"
            raise ValueError(msg)
        self._model.post_processor.image_sensitivity = value

    def detect(self, image_tensor: torch.Tensor) -> Detection:
        """Score one image. Expects (1, 3, H, W) float in [0, 1].

        inference_mode is entered here rather than by the caller because it is
        thread-local, and the API runs this inside a worker thread.
        """
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model(image_tensor)
        duration_ms = int((time.perf_counter() - started) * 1000)

        return Detection(
            category=self.category,
            is_defective=bool(output.pred_label[0]),
            score=float(output.pred_score[0]),
            threshold=self.threshold,
            defect_area_px=int(output.pred_mask[0].sum()) if output.pred_mask is not None else 0,
            duration_ms=duration_ms,
            anomaly_map=output.anomaly_map[0],
            pred_mask=output.pred_mask[0] if output.pred_mask is not None else None,
        )
