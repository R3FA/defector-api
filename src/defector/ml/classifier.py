"""Category classifier: decides which detector an upload should be routed to.

A logistic regression over frozen ResNet18 ImageNet features. Measured at
1.0000 accuracy on the 425-image MVTec test set, including all 304 defective
images, since a damaged part still has to be routed correctly.

When the top probability falls below the configured threshold the result is
None, and the caller is expected to ask the user to pick a category rather than
guessing. scripts/06_ablation_routing.py records why the alternative, comparing
the three PatchCore scores directly, is not used.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from defector.ml.preprocess import classifier_transform
from defector.settings import settings


@dataclass(frozen=True, slots=True)
class CategoryPrediction:
    """category is None when confidence is below the threshold."""

    category: str | None
    confidence: float
    probabilities: dict[str, float]


class CategoryClassifier:
    def __init__(self, bundle_path: Path | None = None) -> None:
        path = bundle_path or settings.classifier_path
        if not path.is_file():
            msg = f"classifier not found at {path}; run scripts/03_train_classifier.py"
            raise FileNotFoundError(msg)

        bundle = joblib.load(path)
        self._model = bundle["classifier"]
        self.categories: list[str] = list(bundle["categories"])
        self.threshold: float = float(
            bundle.get("confidence_threshold", settings.category_confidence_threshold)
        )

        extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        extractor.fc = nn.Identity()
        self._extractor = extractor.eval()
        self._transform = classifier_transform()

    @torch.inference_mode()
    def _features(self, image: Image.Image) -> np.ndarray:
        batch = self._transform(image).unsqueeze(0)
        return self._extractor(batch).numpy()

    def predict(self, image: Image.Image) -> CategoryPrediction:
        probabilities = self._model.predict_proba(self._features(image))[0]
        best = int(probabilities.argmax())
        confidence = float(probabilities[best])
        return CategoryPrediction(
            category=self.categories[best] if confidence >= self.threshold else None,
            confidence=confidence,
            probabilities={
                c: float(p) for c, p in zip(self.categories, probabilities, strict=True)
            },
        )
