"""Evaluation results produced by the training scripts."""

import json

from fastapi import APIRouter, HTTPException, status

from defector.settings import settings

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("")
def get_metrics() -> dict:
    """Everything the evaluation scripts produced.

    Read from disk rather than recomputed: these come from scripts that need the
    dataset, which the running service does not.
    """
    payload: dict[str, object] = {}
    for key, filename in (
        ("detectors", "metrics.json"),
        ("classifier", "classifier.json"),
        ("routing_ablation", "ablation.json"),
        ("thresholds", "evaluation.json"),
        ("latency", "latency.json"),
    ):
        path = settings.artifacts_dir / filename
        if path.is_file():
            payload[key] = json.loads(path.read_text())

    if not payload:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="no evaluation artifacts found; run the scripts in scripts/",
        )
    return payload
