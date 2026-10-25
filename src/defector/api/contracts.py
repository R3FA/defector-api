"""Response models.

SQLite has no timezone type. SQLModel's UTCDateTime reads stored values back as
UTC, but a naive value would let a browser read it as local time, a silent
hour-shift, so the serialiser stamps UTC on any value that lacks an offset.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, field_serializer

from defector.data.entities import Analysis


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    category: str
    category_auto: bool
    category_confidence: float | None
    is_defective: bool
    score: float
    threshold: float
    defect_area_px: int
    duration_ms: int
    original_url: str
    heatmap_url: str

    @field_serializer("created_at")
    def _as_utc(self, value: datetime) -> str:
        """Stamp UTC on naive values read back from SQLite."""
        aware = value if value.tzinfo else value.replace(tzinfo=UTC)
        return aware.isoformat()

    @classmethod
    def of(cls, row: Analysis) -> "AnalysisOut":
        return cls(
            id=row.id,
            created_at=row.created_at,
            category=row.category,
            category_auto=row.category_auto,
            category_confidence=row.category_confidence,
            is_defective=row.is_defective,
            score=row.score,
            threshold=row.threshold,
            defect_area_px=row.defect_area_px,
            duration_ms=row.duration_ms,
            original_url=f"/api/analyses/{row.id}/original",
            heatmap_url=f"/api/analyses/{row.id}/heatmap",
        )


class AnalysisPage(BaseModel):
    items: list[AnalysisOut]
    total: int
    limit: int
    offset: int


class CategoryInfo(BaseModel):
    name: str
    threshold: float


class CategoriesOut(BaseModel):
    categories: list[CategoryInfo]
    confidence_threshold: float


class CategoryStats(BaseModel):
    total: int
    defective: int
    rate: float


class DailyPoint(BaseModel):
    day: str
    total: int
    defective: int
    rate: float


class StatsOut(BaseModel):
    by_category: dict[str, CategoryStats]
    daily: list[DailyPoint]


class AmbiguousCategory(BaseModel):
    """Returned with 409 when the classifier is not confident enough.

    The client is expected to re-post with an explicit category, which is the
    manual fallback the specification allows for.
    """

    detail: str
    confidence: float
    probabilities: dict[str, float]
    categories: list[str]


class AmbiguousCategoryResponse(BaseModel):
    """The 409 body as sent: HTTPException wraps the payload in "detail"."""

    detail: AmbiguousCategory
