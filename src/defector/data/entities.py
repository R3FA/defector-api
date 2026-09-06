"""Database tables.

One row per analysis. Paths are stored as bare filenames, not absolute paths,
so the database survives the project directory being moved: the directories
themselves come from settings.

The threshold in force at the time is stored per row rather than looked up
globally. Sensitivity can be retuned later, and old verdicts still have to be
explainable against the boundary they were actually judged by.
"""

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


class Analysis(SQLModel, table=True):
    __tablename__ = "analysis"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=_now, index=True)

    # Routing
    category: str = Field(index=True)
    category_auto: bool = Field(default=True)
    category_confidence: float | None = Field(default=None)

    # Verdict
    is_defective: bool = Field(index=True)
    score: float
    threshold: float
    defect_area_px: int = Field(default=0)

    # Bookkeeping
    duration_ms: int = Field(default=0)
    original_filename: str
    heatmap_filename: str
