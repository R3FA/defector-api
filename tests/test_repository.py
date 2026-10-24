"""History queries: filters, paging and the derived statistics."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, SQLModel

from defector.data import repository as repo
from defector.data.entities import Analysis


@pytest.fixture
def session() -> Session:
    import defector.data.database as database

    SQLModel.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        for row in session.exec(__import__("sqlmodel").select(Analysis)).all():
            session.delete(row)
        session.commit()

        now = datetime.now(UTC)
        # screw: 2 defective, 3 good.  metal_nut: 1 defective, 4 good.
        for category, defective, good in (("screw", 2, 3), ("metal_nut", 1, 4)):
            for index in range(defective + good):
                session.add(
                    Analysis(
                        created_at=now - timedelta(minutes=index),
                        category=category,
                        is_defective=index < defective,
                        score=0.9 if index < defective else 0.4,
                        threshold=0.5,
                        defect_area_px=100 if index < defective else 0,
                        duration_ms=20,
                        original_filename=f"{category}_{index}.png",
                        heatmap_filename=f"{category}_{index}.png",
                    )
                )
        session.commit()
        yield session


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, 10),
        ({"category": "screw"}, 5),
        ({"category": "metal_nut"}, 5),
        ({"defective": True}, 3),
        ({"defective": False}, 7),
        ({"category": "screw", "defective": True}, 2),
        ({"category": "screw", "defective": False}, 3),
        ({"category": "nepostoji"}, 0),
    ],
)
def test_filters_combine(session: Session, filters: dict, expected: int) -> None:
    assert repo.list_analyses(session, **filters).total == expected


def test_paging_does_not_overlap_and_is_newest_first(session: Session) -> None:
    first = repo.list_analyses(session, limit=4, offset=0)
    second = repo.list_analyses(session, limit=4, offset=4)

    assert len(first.items) == 4
    assert not {row.id for row in first.items} & {row.id for row in second.items}
    timestamps = [row.created_at for row in first.items]
    assert timestamps == sorted(timestamps, reverse=True)


def test_offset_past_the_end_returns_nothing_but_keeps_the_total(session: Session) -> None:
    page = repo.list_analyses(session, limit=10, offset=999)
    assert page.items == []
    assert page.total == 10


def test_category_summary_rates(session: Session) -> None:
    summary = repo.category_summary(session)
    assert summary["screw"] == {"total": 5, "defective": 2, "rate": 0.4}
    assert summary["metal_nut"] == {"total": 5, "defective": 1, "rate": 0.2}


def test_get_returns_none_for_a_missing_id(session: Session) -> None:
    assert repo.get(session, 10**9) is None


def test_daily_rates_cover_the_seeded_rows(session: Session) -> None:
    rates = repo.daily_rates(session, days=30)
    assert sum(day.total for day in rates) == 10
    assert sum(day.defective for day in rates) == 3
