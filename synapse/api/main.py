"""Synapse API (Phase 5) — FastAPI app over the KnowledgeEngine.

Run:  uvicorn synapse.api.main:app --port 8848

The lifespan connects ONE KnowledgeEngine (Neo4j + Ollama + Claude) and exposes it
on app.state.engine; routes read it via the get_engine dependency. Unit tests skip
the lifespan (plain TestClient, no context manager) and override get_engine with a fake.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from synapse.api.routes import capture, curation, graph, knowledge, projects, search, timeline
from synapse.api import websocket
from synapse.api.deps import require_api_key
from synapse.config import settings

# Ensure INFO logs reach the console when the app is loaded by uvicorn directly
# (uvicorn configures its own root logger only after import; we guard to avoid
# double-adding handlers if basicConfig was already called by the process).
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

logger = logging.getLogger("synapse.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    from synapse.core.knowledge_engine import KnowledgeEngine

    # Config safety warnings (spec item 8)
    if not settings.neo4j_password:
        logger.warning("NEO4J_PASSWORD is not set — Neo4j connections will fail")
    if not settings.projects_root:
        logger.warning("SYNAPSE_PROJECTS_ROOT is not set — project connector features are disabled")
    elif not Path(settings.projects_root).exists():
        logger.warning(
            "projects_root does not exist: %s — project connector will fail",
            settings.projects_root,
        )

    logger.info("API starting; connecting knowledge engine...")
    engine = await KnowledgeEngine().connect()
    app.state.engine = engine
    try:
        yield
    finally:
        await engine.close()
        logger.info("API stopped.")


app = FastAPI(title="Synapse API", version="1.0", lifespan=lifespan)

# ── Global exception handler (spec item 1) ────────────────────────────────────
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# ── CORS (spec item 6) ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",   # Dockerized UI
        "http://127.0.0.1:5174",
    ],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


_API = "/api/v1"
_auth = [Depends(require_api_key)]

app.include_router(graph.router,      prefix=_API, dependencies=_auth)
app.include_router(knowledge.router,  prefix=_API, dependencies=_auth)
app.include_router(search.router,     prefix=_API, dependencies=_auth)
app.include_router(timeline.router,   prefix=_API, dependencies=_auth)
app.include_router(projects.router,   prefix=_API, dependencies=_auth)
app.include_router(curation.router,   prefix=_API, dependencies=_auth)
app.include_router(capture.router,    prefix=_API, dependencies=_auth)
app.include_router(websocket.router)  # /ws (no api prefix, no auth)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=settings.synapse_api_port)


if __name__ == "__main__":
    main()
