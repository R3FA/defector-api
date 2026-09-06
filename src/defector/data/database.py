"""SQLite engine and session handling.

Three pragmas are set per connection, none of which SQLite applies by default:

- WAL, so a reader is not blocked by the writer. Inference takes a few hundred
  milliseconds and holds no transaction, but the history view is read while
  analyses are being written.
- busy_timeout, so a concurrent write waits instead of raising
  "database is locked" immediately.
- foreign_keys, which SQLite parses but ignores unless switched on.

SQLAlchemy 2.0 already sets check_same_thread=False for file-based SQLite and
uses QueuePool, so the connect_args seen in older tutorials are unnecessary.
"""

from collections.abc import Iterator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from defector.settings import settings

engine = create_engine(f"sqlite:///{settings.database_path}", echo=False)


@event.listens_for(Engine, "connect")
def _apply_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_tables() -> None:
    """Called once from the application lifespan, not at import time."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with Session(engine) as session:
        yield session
