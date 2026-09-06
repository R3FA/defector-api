#!/usr/bin/env python
"""Train one PatchCore anomaly detector per part category.

Runs outside the web application, as the thesis specification requires. Only
defect-free images from train/good are used; the test split supplies both good
and defective samples with their ground truth masks.

Each category gets its own model and its own Lightning checkpoint. Test metrics
are written to artifacts/metrics.json so the API and the thesis can read the
same numbers without retraining.

Usage:
    python scripts/02_train_detectors.py                     # all categories
    python scripts/02_train_detectors.py screw metal_nut     # a subset
"""

import json
import sys
import time

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator, F1Max, F1Score
from anomalib.models import Patchcore
from rich.console import Console

from defector import threads
from defector.settings import settings

console = Console(width=100)


def build_evaluator() -> Evaluator:
    """Metrics required by the thesis. The default evaluator omits AUPRO and F1Max.

    Pixel metrics need ground truth masks, which exist in the test split but not
    at inference time, so they are registered with strict=False.
    """
    return Evaluator(
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Max(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
            F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
            AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        ],
    )


def train_one(category: str) -> dict[str, object]:
    console.rule(f"[bold]{category}")

    datamodule = MVTecAD(
        root=settings.mvtec_dir,
        category=category,
        train_batch_size=settings.train_batch_size,
        eval_batch_size=settings.eval_batch_size,
        num_workers=settings.num_workers,
        seed=settings.seed,
    )

    # Image size belongs to the pre-processor, not the datamodule, in anomalib 2.x.
    # The same geometry must be used everywhere or feature distances stop being
    # comparable across models.
    pre_processor = Patchcore.configure_pre_processor(
        image_size=settings.image_size,
        center_crop_size=settings.center_crop_size,
    )

    model = Patchcore(
        backbone=settings.backbone,
        layers=tuple(settings.layers),
        pre_trained=True,
        coreset_sampling_ratio=settings.coreset_sampling_ratio,
        num_neighbors=settings.num_neighbors,
        pre_processor=pre_processor,
        evaluator=build_evaluator(),
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir=settings.category_results_dir(category),
        logger=False,
    )

    started = time.time()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = time.time() - started

    started = time.time()
    results = engine.test(model=model, datamodule=datamodule)
    test_seconds = time.time() - started

    metrics = dict(results[0]) if results else {}
    checkpoint = settings.checkpoint_path(category)

    console.print(f"  fit  {fit_seconds / 60:6.1f} min")
    console.print(f"  test {test_seconds / 60:6.1f} min")
    for key in sorted(metrics):
        console.print(f"  {key:24} {metrics[key]:.4f}")
    if not checkpoint.is_file():
        console.print(f"  [red]checkpoint not found at {checkpoint}[/red]")

    return {
        "category": category,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "fit_seconds": round(fit_seconds, 1),
        "test_seconds": round(test_seconds, 1),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
    }


def main(argv: list[str]) -> int:
    categories = argv or list(settings.categories)
    unknown = [c for c in categories if c not in settings.categories]
    if unknown:
        console.print(f"[red]Unknown categories: {unknown}[/red]")
        console.print(f"Known: {list(settings.categories)}")
        return 2

    if not settings.mvtec_dir.is_dir():
        console.print(f"[red]Dataset missing: {settings.mvtec_dir}[/red]")
        console.print("Run scripts/01_check_data.py first.")
        return 1

    settings.ensure_dirs()

    # Explicit rather than relying on import order. Idempotent, and it must run
    # before any tensor work, which starts in train_one().
    threads.apply()
    console.print(f"threads    {threads.describe()}")
    console.print(f"backbone   {settings.backbone}  layers {tuple(settings.layers)}")
    console.print(
        f"coreset    {settings.coreset_sampling_ratio}  neighbors {settings.num_neighbors}"
    )
    console.print(f"geometry   resize {settings.image_size}  crop {settings.center_crop_size}")

    runs = [train_one(category) for category in categories]

    report = {
        "config": {
            "backbone": settings.backbone,
            "layers": list(settings.layers),
            "coreset_sampling_ratio": settings.coreset_sampling_ratio,
            "num_neighbors": settings.num_neighbors,
            "image_size": list(settings.image_size),
            "center_crop_size": list(settings.center_crop_size),
            "accelerator": "cpu",
        },
        "runs": runs,
    }
    output = settings.artifacts_dir / "metrics.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    console.rule()
    console.print(f"[green]Wrote {output.relative_to(settings.artifacts_dir.parent)}[/green]")

    return 0 if all(r["checkpoint_exists"] for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
