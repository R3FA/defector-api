#!/usr/bin/env python
"""Threshold analysis for the trained detectors.

02_train_detectors.py reports threshold-free metrics (AUROC, AUPRO). Those say
how well the score separates good from defective, but they say nothing about
the verdict the application actually returns, which depends on where the
boundary sits.

This script answers the operational question instead. For every test image it
records the score and the true label, then reports the confusion matrix at the
threshold in force, and sweeps the threshold to show what the false positive
and false negative counts trade against each other.

The asymmetry matters. A false positive sends a good part for a second look,
which costs a few seconds. A false negative ships a defective part to the
customer. So the table also reports the most permissive boundary that still
reaches zero false negatives, which is the setting with the fewest false alarms
among those that miss nothing.

Usage:
    python scripts/04_evaluate.py
"""

import json
import sys
import time

import numpy as np
from rich.console import Console
from rich.table import Table

from defector import threads
from defector.ml.detector import Detector
from defector.ml.preprocess import load_tensor
from defector.settings import settings

console = Console(width=110)

# Normalised-score boundaries to sweep. 0.5 is the default, which corresponds
# to image_sensitivity 0.5; a lower boundary is stricter and flags more parts.
SWEEP = [round(0.05 * step, 2) for step in range(2, 19)]


def confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, int]:
    """Positive means defective."""
    predicted = scores >= threshold
    return {
        "tp": int((predicted & (labels == 1)).sum()),
        "fp": int((predicted & (labels == 0)).sum()),
        "tn": int((~predicted & (labels == 0)).sum()),
        "fn": int((~predicted & (labels == 1)).sum()),
    }


def rates(matrix: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = matrix["tp"], matrix["fp"], matrix["tn"], matrix["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "accuracy": round((tp + tn) / max(tp + fp + tn + fn, 1), 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall
        else 0.0,
    }


def score_category(category: str) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """Scores and labels for one category's whole test split."""
    detector = Detector.load(category)
    test_root = settings.mvtec_dir / category / "test"

    scores: list[float] = []
    labels: list[int] = []
    durations: list[float] = []

    for subdirectory in sorted(p for p in test_root.iterdir() if p.is_dir()):
        defective = 0 if subdirectory.name == "good" else 1
        for path in sorted(subdirectory.glob("*.png")):
            tensor, _ = load_tensor(path)
            detection = detector.detect(tensor)
            scores.append(detection.score)
            labels.append(defective)
            durations.append(detection.duration_ms)

    return np.asarray(scores), np.asarray(labels), detector.threshold, durations


def main() -> int:
    if not settings.mvtec_dir.is_dir():
        console.print(f"[red]Dataset missing: {settings.mvtec_dir}[/red]")
        return 1

    settings.ensure_dirs()
    threads.apply()
    console.print(f"backbone {settings.backbone}  coreset {settings.coreset_sampling_ratio}\n")

    report: dict[str, object] = {
        "backbone": settings.backbone,
        "coreset_sampling_ratio": settings.coreset_sampling_ratio,
        "categories": {},
    }

    at_threshold = Table(title="verdicts at the threshold in force")
    at_threshold.add_column("category")
    at_threshold.add_column("thr", justify="right")
    at_threshold.add_column("TP", justify="right")
    at_threshold.add_column("FP", justify="right")
    at_threshold.add_column("TN", justify="right")
    at_threshold.add_column("FN", justify="right")
    at_threshold.add_column("recall", justify="right")
    at_threshold.add_column("precision", justify="right")
    at_threshold.add_column("accuracy", justify="right")

    zero_fn = Table(title="highest boundary that still misses nothing")
    zero_fn.add_column("category")
    zero_fn.add_column("boundary", justify="right")
    zero_fn.add_column("sensitivity", justify="right")
    zero_fn.add_column("FN", justify="right")
    zero_fn.add_column("FP", justify="right")
    zero_fn.add_column("false alarm rate", justify="right")

    started = time.time()
    for category in settings.categories:
        console.print(f"scoring {category} ...")
        scores, labels, threshold, durations = score_category(category)

        default = confusion(scores, labels, threshold)
        default_rates = rates(default)
        at_threshold.add_row(
            category,
            f"{threshold:.2f}",
            str(default["tp"]),
            str(default["fp"]),
            str(default["tn"]),
            str(default["fn"]),
            f"{default_rates['recall']:.4f}",
            f"{default_rates['precision']:.4f}",
            f"{default_rates['accuracy']:.4f}",
        )

        sweep = []
        for boundary in SWEEP:
            matrix = confusion(scores, labels, boundary)
            sweep.append({"threshold": boundary, **matrix, **rates(matrix)})

        # A higher boundary is more permissive and produces fewer false alarms,
        # so among the settings that miss nothing the highest one is the cheapest.
        perfect = [entry for entry in sweep if entry["fn"] == 0]
        best = max(perfect, key=lambda e: e["threshold"]) if perfect else None
        if best:
            negatives = best["fp"] + best["tn"]
            zero_fn.add_row(
                category,
                f"{best['threshold']:.2f}",
                f"{1 - best['threshold']:.2f}",
                "0",
                str(best["fp"]),
                f"{best['fp'] / negatives:.1%}" if negatives else "-",
            )
        else:
            zero_fn.add_row(category, "-", "-", "not reachable in the sweep", "-", "-")

        report["categories"][category] = {
            "threshold": round(threshold, 4),
            "images": len(labels),
            "good": int((labels == 0).sum()),
            "defective": int((labels == 1).sum()),
            "at_threshold": {**default, **default_rates},
            "zero_false_negative": best,
            "sweep": sweep,
            "latency_ms": {
                "mean": round(float(np.mean(durations)), 1),
                "p50": round(float(np.percentile(durations, 50)), 1),
                "p95": round(float(np.percentile(durations, 95)), 1),
                "max": int(np.max(durations)),
            },
        }

    console.print()
    console.print(at_threshold)
    console.print()
    console.print(zero_fn)
    console.print(
        "\n[dim]A false positive costs a re-inspection. A false negative ships a defective "
        "part. The second table is the setting that eliminates the second kind.[/dim]"
    )

    report["seconds"] = round(time.time() - started, 1)
    output = settings.artifacts_dir / "evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    console.print(f"\nsaved {output.relative_to(settings.artifacts_dir.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
