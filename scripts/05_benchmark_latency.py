#!/usr/bin/env python
"""Break down where processing time goes.

The detector alone takes about 140 ms with wide_resnet50_2, but an end-to-end
request takes two to three times that. This measures each stage separately so the number
reported in the thesis is attributed rather than guessed, and so any future
optimisation targets the stage that actually dominates.

Stages measured per image:

  decode      read the file and convert to a tensor
  classify    ResNet18 pass plus the logistic regression
  detect      PatchCore: backbone, nearest-neighbour search, anomaly map
  render      colour-map the heatmap, resize, blend and write the PNG

A warm-up pass runs first and is discarded. The first inference pays for lazy
kernel initialisation in PyTorch and would otherwise skew the mean.

Usage:
    python scripts/05_benchmark_latency.py [repeats]
"""

import json
import sys
import time

import numpy as np
from rich.console import Console
from rich.table import Table

from defector import threads
from defector.ml import overlay
from defector.ml.classifier import CategoryClassifier
from defector.ml.detector import Detector
from defector.ml.preprocess import load_tensor
from defector.settings import settings

console = Console(width=100)
WARMUP = 3


def sample_paths(category: str, count: int) -> list:
    """Alternates good and defective images, since resolution and content differ."""
    test_root = settings.mvtec_dir / category / "test"
    good = sorted((test_root / "good").glob("*.png"))
    defect_dirs = sorted(p for p in test_root.iterdir() if p.is_dir() and p.name != "good")
    bad = [path for directory in defect_dirs for path in sorted(directory.glob("*.png"))]

    picked = []
    for index in range(count):
        pool = good if index % 2 == 0 else bad
        picked.append(pool[index % len(pool)])
    return picked


def main(argv: list[str]) -> int:
    repeats = int(argv[0]) if argv else 20

    if not settings.mvtec_dir.is_dir():
        console.print(f"[red]Dataset missing: {settings.mvtec_dir}[/red]")
        return 1

    settings.ensure_dirs()
    threads.apply()
    console.print(f"threads {threads.describe()}")
    console.print(f"{repeats} images per category, {WARMUP} discarded warm-up passes\n")

    load_started = time.perf_counter()
    detectors = {c: Detector.load(c) for c in settings.categories}
    classifier = CategoryClassifier()
    load_seconds = time.perf_counter() - load_started
    console.print(f"model load (once at startup): {load_seconds:.2f} s")

    # Label the report with what the checkpoints were trained with, which can
    # differ from the backbone configured now.
    trained = {(d.backbone, d.coreset_sampling_ratio) for d in detectors.values()}
    if len(trained) != 1:
        console.print(f"[red]checkpoints disagree on their configuration: {trained}[/red]")
        return 1
    backbone, coreset = trained.pop()
    console.print(f"checkpoints: backbone {backbone}  coreset {coreset}\n")

    report: dict[str, object] = {
        "backbone": backbone,
        "coreset_sampling_ratio": coreset,
        "threads": threads.describe(),
        "repeats_per_category": repeats,
        "model_load_seconds": round(load_seconds, 2),
        "categories": {},
    }

    table = Table(title="milliseconds per stage, mean (p95)")
    table.add_column("category")
    table.add_column("decode", justify="right")
    table.add_column("classify", justify="right")
    table.add_column("detect", justify="right")
    table.add_column("render", justify="right")
    table.add_column("total", justify="right")

    for category in settings.categories:
        detector = detectors[category]
        paths = sample_paths(category, repeats + WARMUP)
        stages: dict[str, list[float]] = {k: [] for k in ("decode", "classify", "detect", "render")}

        for index, path in enumerate(paths):
            mark = time.perf_counter()
            tensor, image = load_tensor(path.read_bytes())
            decode = (time.perf_counter() - mark) * 1000

            mark = time.perf_counter()
            classifier.predict(image)
            classify = (time.perf_counter() - mark) * 1000

            mark = time.perf_counter()
            detection = detector.detect(tensor)
            detect = (time.perf_counter() - mark) * 1000

            mark = time.perf_counter()
            overlay.render(
                detection.anomaly_map,
                image,
                pixel_threshold=detector.pixel_threshold,
                filename=f"bench_{category}.png",
            )
            render = (time.perf_counter() - mark) * 1000

            if index < WARMUP:
                continue
            stages["decode"].append(decode)
            stages["classify"].append(classify)
            stages["detect"].append(detect)
            stages["render"].append(render)

        summary = {}
        for name, values in stages.items():
            array = np.asarray(values)
            summary[name] = {
                "mean": round(float(array.mean()), 1),
                "p50": round(float(np.percentile(array, 50)), 1),
                "p95": round(float(np.percentile(array, 95)), 1),
                "max": round(float(array.max()), 1),
            }

        totals = np.asarray([sum(v) for v in zip(*stages.values(), strict=True)])
        summary["total"] = {
            "mean": round(float(totals.mean()), 1),
            "p50": round(float(np.percentile(totals, 50)), 1),
            "p95": round(float(np.percentile(totals, 95)), 1),
            "max": round(float(totals.max()), 1),
        }
        report["categories"][category] = summary

        table.add_row(
            category,
            *(
                f"{summary[name]['mean']:.0f} ({summary[name]['p95']:.0f})"
                for name in ("decode", "classify", "detect", "render", "total")
            ),
        )

    console.print(table)

    # Which stage dominates, averaged over categories.
    shares: dict[str, float] = {}
    for name in ("decode", "classify", "detect", "render"):
        shares[name] = float(
            np.mean([report["categories"][c][name]["mean"] for c in settings.categories])
        )
    grand = sum(shares.values())
    console.print("\nshare of total:")
    for name, value in sorted(shares.items(), key=lambda kv: -kv[1]):
        console.print(f"  {name:9} {value:6.1f} ms  {value / grand:5.1%}")
    report["share_of_total"] = {k: round(v / grand, 4) for k, v in shares.items()}

    for leftover in settings.heatmaps_dir.glob("bench_*.png"):
        leftover.unlink()

    output = settings.artifacts_dir / "latency.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    console.print(f"\nsaved {output.relative_to(settings.artifacts_dir.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
