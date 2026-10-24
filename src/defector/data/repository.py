"""Queries over the analysis history.

Covers the history requirements: list past analyses, filter by category and by
outcome, reopen a single result, and report the share of defective parts over
time. Every filter here is backed by an index on the table.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlmodel import Session, func, select

from defector.data.entities import Analysis


@dataclass(frozen=True, slots=True)
class Page:
    items: list[Analysis]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class DailyRate:
    day: date
    total: int
    defective: int

    @property
    def rate(self) -> float:
        return self.defective / self.total if self.total else 0.0


def create(session: Session, analysis: Analysis) -> Analysis:
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    return analysis


def get(session: Session, analysis_id: int) -> Analysis | None:
    return session.get(Analysis, analysis_id)


def list_analyses(
    session: Session,
    *,
    category: str | None = None,
    defective: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> Page:
    """Newest first. Both filters are optional and combine."""
    filters = []
    if category is not None:
        filters.append(Analysis.category == category)
    if defective is not None:
        filters.append(Analysis.is_defective == defective)

    total = session.exec(select(func.count()).select_from(Analysis).where(*filters)).one()

    items = session.exec(
        select(Analysis)
        .where(*filters)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()

    return Page(items=list(items), total=int(total), limit=limit, offset=offset)


def category_summary(session: Session) -> dict[str, dict[str, int | float]]:
    """Counts and defect rate per category."""
    rows = session.exec(
        select(Analysis.category, Analysis.is_defective, func.count()).group_by(
            Analysis.category, Analysis.is_defective
        )
    ).all()

    totals: Counter[str] = Counter()
    defective: Counter[str] = Counter()
    for category, is_defective, count in rows:
        totals[category] += count
        if is_defective:
            defective[category] += count

    return {
        category: {
            "total": totals[category],
            "defective": defective[category],
            "rate": round(defective[category] / totals[category], 4) if totals[category] else 0.0,
        }
        for category in sorted(totals)
    }


def daily_rates(session: Session, *, days: int = 30) -> list[DailyRate]:
    """Defect share per day. Served by /api/analyses/stats; no view charts it yet.

    Grouping is done in Python rather than SQL because SQLite stores naive
    datetimes and date bucketing there would depend on the stored text format.
    """
    since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
    rows = session.exec(
        select(Analysis.created_at, Analysis.is_defective).where(Analysis.created_at >= since)
    ).all()

    totals: Counter[date] = Counter()
    defective: Counter[date] = Counter()
    for created_at, is_defective in rows:
        day = created_at.date()
        totals[day] += 1
        if is_defective:
            defective[day] += 1

    return [
        DailyRate(day=day, total=totals[day], defective=defective[day]) for day in sorted(totals)
    ]
