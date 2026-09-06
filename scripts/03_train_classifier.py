#!/usr/bin/env python
"""Train the part-category classifier used to route an upload to the right detector.

A linear probe on frozen ImageNet features, not a fine-tuned network: a ResNet18
with its classification head removed produces a 512-d vector per image, and a
logistic regression maps that to one of the part categories. Nothing is trained
end to end, so this takes seconds on CPU and adds no dependencies.

Why not compare the three PatchCore models directly and pick the lowest score:
anomalib normalises pred_score per model against that model's own validation
threshold and clamps to [0, 1], so scores from different models are not
comparable and ties at 1.0 are resolved arbitrarily. The memory banks also
differ in density (320 vs 220 vs 224 training images), which biases the
comparison systematically. scripts/06_ablation_routing.py measures this.

Training uses train/good only, but evaluation also covers the defective test
images, because a damaged screw is still a screw and the router must survive it.

Usage:
    python scripts/03_train_classifier.py
"""

import json
import sys
import time

import joblib
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from defector import threads
from defector.settings import settings

console = Console(width=100)
BATCH = 16


def build_extractor() -> tuple[nn.Module, object]:
    """ResNet18 with the classification head removed, plus its matching transforms."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = nn.Identity()
    net.eval()
    return net, weights.transforms()


@torch.inference_mode()
def features_for(paths: list, net: nn.Module, transform) -> np.ndarray:
    """512-d feature per image, batched."""
    from PIL import Image

    out = []
    for start in range(0, len(paths), BATCH):
        chunk = paths[start : start + BATCH]
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in chunk])
        out.append(net(batch).numpy())
    return np.concatenate(out) if out else np.empty((0, 512), dtype=np.float32)


def collect(subset: str) -> tuple[list, list[int]]:
    """Image paths and label indices. subset is 'train' or 'test'."""
    paths: list = []
    labels: list[int] = []
    for index, category in enumerate(settings.categories):
        root = settings.mvtec_dir / category / subset
        if subset == "train":
            found = sorted((root / "good").glob("*.png"))
        else:
            # good and every defect directory: the router must handle both
            found = sorted(p for p in root.rglob("*.png"))
        paths.extend(found)
        labels.extend([index] * len(found))
    return paths, labels


def main() -> int:
    if not settings.mvtec_dir.is_dir():
        console.print(f"[red]Dataset missing: {settings.mvtec_dir}[/red]")
        return 1

    settings.ensure_dirs()
    threads.apply()

    net, transform = build_extractor()
    console.print("extractor  resnet18 (ImageNet), head removed -> 512-d")
    console.print(f"categories {list(settings.categories)}")

    started = time.time()
    train_paths, y_train = collect("train")
    console.print(f"\nextracting features for {len(train_paths)} train/good images ...")
    x_train = features_for(train_paths, net, transform)

    test_paths, y_test = collect("test")
    console.print(f"extracting features for {len(test_paths)} test images (good + defective) ...")
    x_test = features_for(test_paths, net, transform)
    extract_seconds = time.time() - started

    classifier = LogisticRegression(max_iter=2000)
    cv_scores = cross_val_score(classifier, x_train, np.asarray(y_train), cv=5)
    classifier.fit(x_train, y_train)

    probabilities = classifier.predict_proba(x_test)
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    accuracy = float((predictions == np.asarray(y_test)).mean())
    threshold = settings.category_confidence_threshold
    below = int((confidences < threshold).sum())

    # --- report -----------------------------------------------------------
    console.print(f"\nfeature extraction {extract_seconds:.1f}s")
    console.print(f"5-fold CV on train/good  {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")
    console.print(f"accuracy on test set     {accuracy:.4f}   ({len(test_paths)} images)")
    console.print(
        f"below confidence {threshold}   {below} of {len(test_paths)}"
        f"  -> the app would ask the user to choose"
    )

    table = Table(title="confusion matrix on test set", show_lines=False)
    table.add_column("actual \\ predicted")
    for category in settings.categories:
        table.add_column(category, justify="right")
    for row, category in zip(
        confusion_matrix(y_test, predictions), settings.categories, strict=True
    ):
        table.add_row(category, *(str(v) for v in row))
    console.print(table)

    # --- persist ----------------------------------------------------------
    bundle = {
        "classifier": classifier,
        "categories": list(settings.categories),
        "backbone": "resnet18",
        "weights": "IMAGENET1K_V1",
        "feature_dim": int(x_train.shape[1]),
        "confidence_threshold": threshold,
    }
    joblib.dump(bundle, settings.classifier_path)
    console.print(f"\nsaved {settings.classifier_path.relative_to(settings.artifacts_dir.parent)}")

    report = {
        "backbone": "resnet18",
        "feature_dim": int(x_train.shape[1]),
        "train_images": len(train_paths),
        "test_images": len(test_paths),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "test_accuracy": round(accuracy, 4),
        "confidence_threshold": threshold,
        "below_threshold": below,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "extract_seconds": round(extract_seconds, 1),
    }
    output = settings.artifacts_dir / "classifier.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    console.print(f"saved {output.relative_to(settings.artifacts_dir.parent)}")

    return 0 if cv_scores.mean() > 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
