"""Shared fixtures.

Every test runs against an isolated database and storage directory, so a test
run never touches the analyses a developer made by hand.
"""

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture(scope="session", autouse=True)
def _isolate_storage(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point settings at a throwaway directory before anything imports the app."""
    root = tmp_path_factory.mktemp("defector")

    from defector.settings import settings

    settings.database_path = root / "test.db"
    settings.uploads_dir = root / "uploads"
    settings.heatmaps_dir = root / "heatmaps"
    settings.artifacts_dir = root / "artifacts"
    settings.ensure_dirs()

    # The engine reads database_path at import time, so it is rebuilt here.
    from sqlmodel import create_engine

    import defector.data.database as database

    database.engine = create_engine(f"sqlite:///{settings.database_path}", echo=False)

    yield root


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="JPEG")
    return buffer.getvalue()
