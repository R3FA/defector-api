"""FastAPI application.

Models are loaded once in the lifespan handler, not per request: three PatchCore
checkpoints, each with its own copy of the backbone, take about four seconds to
load (3.77 s measured) and are then resident. on_event is deprecated in FastAPI and is not used, even though the
SQLModel documentation still shows it.

Run with a single worker. Each additional worker process would execute this
lifespan again and hold its own copy of all three memory banks, for no gain,
since inference is already bounded by the semaphore in deps.py.
"""

# Applies the PyTorch thread caps. Imported first, before anything pulls torch.
import defector.threads  # noqa: F401  # isort: skip

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from defector.api.routes import analyses, analyze, categories, metrics
from defector.data.database import create_tables
from defector.ml.registry import registry
from defector.settings import settings

logger = logging.getLogger(__name__)

# The Angular dev server. In production the built frontend is served from the
# same origin and this is unnecessary.
DEV_ORIGINS = ["http://localhost:4200", "http://127.0.0.1:4200"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    create_tables()
    logger.info("loading models ...")
    registry.load()
    logger.info("ready: %s", registry.categories)
    yield


app = FastAPI(
    title="defector-api",
    description=(
        "Automated visual quality control for machine parts using unsupervised "
        "anomaly detection (PatchCore, MVTec AD)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(analyses.router)
app.include_router(metrics.router)
app.include_router(categories.router)


@app.get("/api/health", tags=["health"])
def health() -> dict[str, object]:
    return {"status": "ok" if registry.ready else "loading", "categories": registry.categories}
