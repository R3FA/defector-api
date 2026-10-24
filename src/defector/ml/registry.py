"""Holds every model in memory and turns an upload into a finished analysis.

The web application never trains and never loads a checkpoint per request. This
registry is built once during application startup, keeps the three PatchCore
detectors and the category classifier resident, and exposes a single analyse()
call.

The three wide_resnet50_2 checkpoints come to about 320 MB, each carrying its
own copy of the backbone, which is why the API must run with a single worker:
every additional worker process would hold its own copy of all three.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from defector.ml.classifier import CategoryClassifier, CategoryPrediction
from defector.ml.detector import Detection, Detector
from defector.ml.overlay import render
from defector.settings import settings


class UnknownCategoryError(Exception):
    """The classifier was not confident enough and no category was supplied.

    The caller should ask the user to choose rather than guessing, which the
    thesis specification allows for explicitly.
    """

    def __init__(self, prediction: CategoryPrediction) -> None:
        super().__init__(
            f"category confidence {prediction.confidence:.4f} is below "
            f"{settings.category_confidence_threshold}"
        )
        self.prediction = prediction


@dataclass(frozen=True, slots=True)
class Analysis:
    """Everything one upload produces."""

    category: str
    category_auto: bool
    category_confidence: float | None
    detection: Detection
    heatmap_path: Path


class ModelRegistry:
    def __init__(self) -> None:
        self._detectors: dict[str, Detector] = {}
        self._classifier: CategoryClassifier | None = None

    def load(self) -> None:
        """Called once at startup. Raises if any artifact is missing."""
        self._detectors = {c: Detector.load(c) for c in settings.categories}
        self._classifier = CategoryClassifier()

    @property
    def ready(self) -> bool:
        return bool(self._detectors) and self._classifier is not None

    @property
    def categories(self) -> list[str]:
        return list(self._detectors)

    def detector(self, category: str) -> Detector:
        try:
            return self._detectors[category]
        except KeyError:
            msg = f"unknown category {category!r}; known: {self.categories}"
            raise KeyError(msg) from None

    def thresholds(self) -> dict[str, float]:
        return {c: d.threshold for c, d in self._detectors.items()}

    def classify(self, image: Image.Image) -> CategoryPrediction:
        if self._classifier is None:
            msg = "registry not loaded"
            raise RuntimeError(msg)
        return self._classifier.predict(image)

    def analyse(
        self,
        image_tensor,
        image: Image.Image,
        *,
        category: str | None = None,
        heatmap_name: str | None = None,
    ) -> Analysis:
        """Route, score and render. Blocking and CPU-bound by design.

        The API calls this from a worker thread, so no async here.
        """
        if category is not None:
            resolved, auto, confidence = category, False, None
        else:
            prediction = self.classify(image)
            if prediction.category is None:
                raise UnknownCategoryError(prediction)
            resolved, auto, confidence = prediction.category, True, prediction.confidence

        detector = self.detector(resolved)
        detection = detector.detect(image_tensor)

        heatmap_path = render(
            detection.anomaly_map,
            image,
            pixel_threshold=detector.pixel_threshold,
            filename=heatmap_name,
        )

        return Analysis(
            category=resolved,
            category_auto=auto,
            category_confidence=confidence,
            detection=detection,
            heatmap_path=heatmap_path,
        )


registry = ModelRegistry()
