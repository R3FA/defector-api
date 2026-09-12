"""Upload an image and get a verdict."""

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session

from defector.api.contracts import AmbiguousCategory, AnalysisOut
from defector.api.deps import inference_slots
from defector.data.database import get_session
from defector.data.entities import Analysis
from defector.data.repository import create
from defector.ml.preprocess import to_tensor
from defector.ml.registry import UnknownCategoryError, registry
from defector.security.upload import UploadRejected, ValidatedUpload, validate
from defector.settings import settings

logger = logging.getLogger(__name__)
SessionDep = Annotated[Session, Depends(get_session)]

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


def _run(upload: ValidatedUpload, category: str | None) -> tuple[Analysis, list[Path]]:
    """Blocking: routing, inference, rendering, and both files saved.

    Executed in a worker thread. torch.inference_mode is thread-local, so it is
    entered inside Detector.detect rather than out here.

    Analysis runs before anything is written, so an upload the classifier cannot
    route leaves nothing behind. The paths are returned so the caller can remove
    them if the database insert then fails; otherwise a crash between writing
    the files and committing the row would orphan both with nothing referencing
    them.
    """
    analysis = registry.analyse(
        to_tensor(upload.image),
        upload.image,
        category=category,
        heatmap_name=f"{upload.stem}.png",
    )
    detection = analysis.detection

    original_path = settings.uploads_dir / upload.filename
    upload.image.save(original_path)
    written = [original_path, analysis.heatmap_path]

    return Analysis(
        category=analysis.category,
        category_auto=analysis.category_auto,
        category_confidence=analysis.category_confidence,
        is_defective=detection.is_defective,
        score=detection.score,
        threshold=detection.threshold,
        defect_area_px=detection.defect_area_px,
        duration_ms=detection.duration_ms,
        original_filename=upload.filename,
        heatmap_filename=analysis.heatmap_path.name,
    ), written


@router.post(
    "",
    response_model=AnalysisOut,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": AmbiguousCategory}},
)
async def analyze(
    session: SessionDep,
    file: Annotated[UploadFile, File(description="Photograph of a machine part")],
    category: Annotated[
        str | None, Form(description="Optional. Skips automatic routing when supplied.")
    ] = None,
) -> AnalysisOut:
    payload = await file.read()

    try:
        upload = validate(payload)
    except UploadRejected as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(error)) from None

    if category is not None and category not in registry.categories:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"unknown category {category!r}; known: {registry.categories}",
        )

    # The semaphore bounds concurrent inference; run_in_threadpool keeps the
    # event loop free while this one runs.
    async with inference_slots:
        try:
            row, written = await run_in_threadpool(_run, upload, category)
        except UnknownCategoryError as error:
            prediction = error.prediction
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "detail": (
                        "the part category could not be determined confidently; "
                        "resubmit with an explicit category"
                    ),
                    "confidence": prediction.confidence,
                    "probabilities": prediction.probabilities,
                    "categories": registry.categories,
                },
            ) from None

    try:
        stored = create(session, row)
    except Exception:
        # The files are referenced by nothing if the row never lands, so remove
        # them rather than leaving them to accumulate.
        for path in written:
            path.unlink(missing_ok=True)
        logger.exception("failed to record analysis; removed %d orphaned file(s)", len(written))
        raise

    return AnalysisOut.of(stored)
