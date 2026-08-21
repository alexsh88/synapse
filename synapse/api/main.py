"""Synapse API (Phase 5) — FastAPI app over the KnowledgeEngine.

Run:  uvicorn synapse.api.main:app --port 8848

The lifespan connects ONE KnowledgeEngine (Neo4j + Ollama + Claude) and exposes it
on app.state.engine; routes read it via the get_engine dependency. Unit tests skip
the lifespan (plain TestClient, no context manager) and override get_engine with a fake.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from synapse.api.routes import capture, curation, graph, knowledge, projects, search, timeline
from synapse.api import websocket
from synapse.api.deps import require_api_key
from synapse.config import settings
from synapse.core import registry

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
    else:
        # Every root, not just the primary: an extra root is typically its own bind mount, and a
        # mount that silently failed to appear looks exactly like "project folder not found".
        for root in registry.project_roots():
            if not root.exists():
                logger.warning(
                    "configured project root does not exist: %s — projects under it cannot connect",
                    root,
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
async def health() -> dict[str, str]:
    """LIVENESS only — "this process is answering". It deliberately touches nothing.

    Kept as-is because the compose healthcheck, the migration runbook and the UI all call it.
    For "can this instance actually serve a request", use /api/v1/readyz — the distinction
    matters, because this endpoint returns ok with Neo4j, Redis and Ollama all unreachable.
    """
    return {"status": "ok"}


@app.get("/api/v1/readyz")
async def readyz() -> JSONResponse:
    """READINESS — reaches every dependency and reports each one by name.

    503 when anything required is down, so `depends_on: service_healthy` and any external
    probe mean "ready to serve" rather than "process started". Redis is optional (it caches
    briefs), so an unconfigured Redis is reported without failing readiness; a *configured*
    Redis that cannot be reached does fail it.
    """
    checks: dict[str, str] = {}
    required: list[str] = ["engine", "neo4j"]

    engine = getattr(app.state, "engine", None)
    if engine is None:
        checks["engine"] = "not connected"
    else:
        checks["engine"] = "ok"
        driver = getattr(engine.graphiti, "driver", None)
        if driver is None:
            checks["neo4j"] = "no driver"
        else:
            try:
                await driver.execute_query("RETURN 1")
                checks["neo4j"] = "ok"
            except Exception as exc:  # noqa: BLE001 — report any failure, never raise from a probe
                checks["neo4j"] = f"error: {type(exc).__name__}"

        redis = getattr(engine.reader, "redis", None) if engine.reader is not None else None
        if redis is None:
            checks["redis"] = "not configured"
        else:
            required.append("redis")
            try:
                await redis.ping()
                checks["redis"] = "ok"
            except Exception as exc:  # noqa: BLE001
                checks["redis"] = f"error: {type(exc).__name__}"

    ready = all(checks.get(name) == "ok" for name in required)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready, "checks": checks},
    )


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
