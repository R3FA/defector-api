"""Analysis history: listing, filtering, single results and the stored images."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from sqlmodel import Session

from defector.api.contracts import (
    AnalysisOut,
    AnalysisPage,
    CategoryStats,
    DailyPoint,
    StatsOut,
)
from defector.data.database import get_session
from defector.data.repository import category_summary, daily_rates, get, list_analyses
from defector.settings import settings

SessionDep = Annotated[Session, Depends(get_session)]

# SQLite stores a signed 64-bit INTEGER. Without an upper bound a larger id
# reaches the driver and raises OverflowError, which surfaces as a 500 instead
# of a validation error. ge=1 also rejects 0 and negatives, which no row can have.
AnalysisId = Annotated[int, PathParam(ge=1, le=2**63 - 1)]

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


@router.get("", response_model=AnalysisPage)
def list_history(
    session: SessionDep,
    category: Annotated[str | None, Query(description="Filter by part category")] = None,
    defective: Annotated[bool | None, Query(description="Filter by outcome")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AnalysisPage:
    page = list_analyses(
        session, category=category, defective=defective, limit=limit, offset=offset
    )
    return AnalysisPage(
        items=[AnalysisOut.of(row) for row in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


# Declared before /{analysis_id} so "stats" is not parsed as an id.
@router.get("/stats", response_model=StatsOut)
def get_stats(
    session: SessionDep,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> StatsOut:
    return StatsOut(
        by_category={
            name: CategoryStats(**values) for name, values in category_summary(session).items()
        },
        daily=[
            DailyPoint(
                day=row.day.isoformat(),
                total=row.total,
                defective=row.defective,
                rate=round(row.rate, 4),
            )
            for row in daily_rates(session, days=days)
        ],
    )


@router.get("/{analysis_id}", response_model=AnalysisOut)
def get_one(analysis_id: AnalysisId, session: SessionDep) -> AnalysisOut:
    row = get(session, analysis_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="analysis not found")
    return AnalysisOut.of(row)


@router.get("/{analysis_id}/original", response_class=FileResponse)
def get_original(analysis_id: AnalysisId, session: SessionDep) -> FileResponse:
    return _serve(session, analysis_id, "original")


@router.get("/{analysis_id}/heatmap", response_class=FileResponse)
def get_heatmap(analysis_id: AnalysisId, session: SessionDep) -> FileResponse:
    return _serve(session, analysis_id, "heatmap")


def _serve(session: Session, analysis_id: int, kind: str) -> FileResponse:
    """FileResponse adds Content-Length, Last-Modified and ETag, which let the
    browser cache the images heuristically. It does not answer a conditional
    request with 304, so an expired entry is downloaded again in full.

    Cache-Control immutable would be wrong here: SQLite reuses row ids after the
    database is reset, and a cached image would then show under the wrong id."""
    row = get(session, analysis_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="analysis not found")

    if kind == "original":
        path = settings.uploads_dir / row.original_filename
    else:
        path = settings.heatmaps_dir / row.heatmap_filename

    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"{kind} image is missing on disk")
    return FileResponse(path)
