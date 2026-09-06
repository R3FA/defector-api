#!/usr/bin/env python
"""Measure whether the three PatchCore models can route an upload to its category.

The tempting shortcut is to skip a classifier entirely: run the image through
all three detectors and keep the one with the lowest anomaly score. This script
quantifies how well that actually works, in four variants, and compares them
against the linear probe in 03_train_classifier.py.

  A  argmin over the normalised pred_score, i.e. what a naive implementation
     using the public API would do
  B  argmin over the raw pred_score, bypassing the post-processor
  C  argmin over the mean raw nearest-neighbour distance per patch
  D  argmin over the same mean distance, z-calibrated per model against that
     model's own train/good distribution

Two reasons A is expected to fail. Anomalib normalises each score against that
model's own validation threshold and min/max, which are unrelated between
models, then clamps to [0, 1]; out-of-category images saturate at 1.0 and the
argmin picks arbitrarily. And the memory banks differ in density, because the
categories have 320, 220 and 224 training images, which biases the comparison
in a fixed direction rather than randomly.

All three models share identical frozen backbone weights, since PatchCore never
updates them, so one backbone pass per image is reused across all three banks.
The script asserts this rather than assuming it.

Usage:
    python scripts/06_ablation_routing.py
"""

import csv
import json
import sys
import time

import numpy as np
import torch
from anomalib.data.utils import read_image
from anomalib.models import Patchcore
from rich.console import Console
from rich.table import Table

from defector import threads
from defector.settings import settings

console = Console(width=100)
CALIBRATION_SAMPLES = 40


def load_models() -> dict[str, Patchcore]:
    models = {}
    for category in settings.categories:
        checkpoint = settings.checkpoint_path(category)
        if not checkpoint.is_file():
            console.print(f"[red]Missing checkpoint for {category}: {checkpoint}[/red]")
            console.print("Run scripts/02_train_detectors.py first.")
            raise SystemExit(1)
        model = Patchcore.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)
        models[category] = model.eval()
    return models


def assert_shared_backbone(models: dict[str, Patchcore]) -> None:
    """Sharing one backbone pass is only valid if the weights are byte-identical."""
    reference = None
    for category, model in models.items():
        state = model.model.feature_extractor.state_dict()
        digest = tuple(float(v.double().sum()) for v in state.values())
        if reference is None:
            reference = digest
        elif digest != reference:
            msg = f"backbone weights differ for {category}; cannot share the forward pass"
            raise SystemExit(msg)
    console.print(f"backbone weights identical across {len(models)} models, pass is shared")


@torch.inference_mode()
def embed(image_path, model: Patchcore) -> tuple[torch.Tensor, int, int]:
    """One backbone pass. Returns flat patch embeddings plus the patch grid."""
    raw = read_image(image_path, as_tensor=True).unsqueeze(0)
    pre = model.pre_processor(raw).type(model.model.memory_bank.dtype)
    features = model.model.feature_extractor(pre)
    features = {k: model.model.feature_pooler(v) for k, v in features.items()}
    grid = model.model.generate_embedding(features)
    _, _, width, height = grid.shape
    return model.model.reshape_embedding(grid), width, height


@torch.inference_mode()
def score_against(embedding: torch.Tensor, model: Patchcore) -> tuple[float, float, float]:
    """Raw pred_score, mean raw distance, and the analytically normalised score.

    Mirrors PatchcoreModel.forward: nearest_neighbors is queried with
    n_neighbors=1, and num_neighbors is used inside compute_anomaly_score for
    the density reweighting.
    """
    patch_scores, locations = model.model.nearest_neighbors(embedding=embedding, n_neighbors=1)
    reshaped_scores = patch_scores.reshape((1, -1))
    reshaped_locations = locations.reshape((1, -1))
    raw = float(model.model.compute_anomaly_score(reshaped_scores, reshaped_locations, embedding))
    mean_distance = float(patch_scores.mean())

    post = model.post_processor
    span = float(post.image_max) - float(post.image_min)
    normalised = ((raw - float(post.image_threshold)) / span) + 0.5 if span else raw
    return raw, mean_distance, min(max(normalised, 0.0), 1.0)


def calibrate(models: dict[str, Patchcore]) -> dict[str, tuple[float, float]]:
    """Mean and std of the in-category mean distance, from train/good."""
    stats = {}
    for category, model in models.items():
        paths = sorted((settings.mvtec_dir / category / "train" / "good").glob("*.png"))
        sample = paths[:CALIBRATION_SAMPLES]
        distances = []
        for path in sample:
            embedding, _, _ = embed(path, model)
            _, mean_distance, _ = score_against(embedding, model)
            distances.append(mean_distance)
        array = np.asarray(distances)
        stats[category] = (float(array.mean()), float(array.std()) or 1.0)
        console.print(
            f"  {category:10} mu {array.mean():7.4f}  sigma {array.std():7.4f}  (n={len(sample)})"
        )
    return stats


def main() -> int:
    if not settings.mvtec_dir.is_dir():
        console.print(f"[red]Dataset missing: {settings.mvtec_dir}[/red]")
        return 1

    settings.ensure_dirs()
    threads.apply()

    models = load_models()
    assert_shared_backbone(models)
    reference_model = models[settings.categories[0]]

    console.print("\ncalibrating on train/good")
    stats = calibrate(models)

    console.print("\nscoring the test set against all three banks")
    started = time.time()
    rows: list[dict[str, object]] = []
    for truth in settings.categories:
        for path in sorted((settings.mvtec_dir / truth / "test").rglob("*.png")):
            embedding, _, _ = embed(path, reference_model)
            row: dict[str, object] = {
                "image": f"{truth}/{path.parent.name}/{path.name}",
                "true_category": truth,
                "is_defective": path.parent.name != "good",
            }
            for category, model in models.items():
                raw, mean_distance, normalised = score_against(embedding, model)
                mu, sigma = stats[category]
                row[f"norm_{category}"] = round(normalised, 6)
                row[f"raw_{category}"] = round(raw, 6)
                row[f"dist_{category}"] = round(mean_distance, 6)
                row[f"z_{category}"] = round((mean_distance - mu) / sigma, 6)
            rows.append(row)
    elapsed = time.time() - started
    console.print(f"  {len(rows)} images in {elapsed / 60:.1f} min")

    # --- evaluate the four strategies -------------------------------------
    categories = list(settings.categories)
    strategies = {
        "A normalised pred_score": "norm",
        "B raw pred_score": "raw",
        "C mean raw distance": "dist",
        "D z-calibrated distance": "z",
    }

    table = Table(title="routing accuracy on the 425-image test set")
    table.add_column("strategy")
    table.add_column("accuracy", justify="right")
    table.add_column("errors", justify="right")
    table.add_column("ties", justify="right")

    summary: dict[str, dict[str, object]] = {}
    for label, prefix in strategies.items():
        correct = ties = 0
        for row in rows:
            values = [row[f"{prefix}_{c}"] for c in categories]
            lowest = min(values)
            winners = [c for c, v in zip(categories, values, strict=True) if v == lowest]
            if len(winners) > 1:
                ties += 1
            if winners[0] == row["true_category"]:
                correct += 1
        accuracy = correct / len(rows)
        summary[label] = {
            "accuracy": round(accuracy, 4),
            "errors": len(rows) - correct,
            "ties": ties,
        }
        table.add_row(label, f"{accuracy:.4f}", str(len(rows) - correct), str(ties))

    # the linear probe actually shipped, for comparison
    probe_path = settings.artifacts_dir / "classifier.json"
    probe_accuracy = None
    if probe_path.is_file():
        probe_accuracy = json.loads(probe_path.read_text())["test_accuracy"]
        table.add_row(
            "linear probe (shipped)",
            f"{probe_accuracy:.4f}",
            str(round((1 - probe_accuracy) * len(rows))),
            "0",
        )

    console.print()
    console.print(table)

    saturated = sum(1 for row in rows if sum(1 for c in categories if row[f"norm_{c}"] >= 1.0) >= 2)
    console.print(
        f"\nimages where two or more models saturate the normalised score at 1.0: "
        f"[bold]{saturated}[/bold] of {len(rows)}"
    )
    console.print("that is the clamp in the post-processor, and it is why strategy A ties")

    # --- persist ----------------------------------------------------------
    csv_path = settings.artifacts_dir / "ablation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = settings.artifacts_dir / "ablation.json"
    json_path.write_text(
        json.dumps(
            {
                "images": len(rows),
                "calibration_samples": CALIBRATION_SAMPLES,
                "calibration": {k: {"mu": v[0], "sigma": v[1]} for k, v in stats.items()},
                "strategies": summary,
                "linear_probe_accuracy": probe_accuracy,
                "saturated_at_one": saturated,
                "seconds": round(elapsed, 1),
            },
            indent=2,
        )
        + "\n"
    )
    console.print(f"\nsaved {csv_path.name} and {json_path.name} in artifacts/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
