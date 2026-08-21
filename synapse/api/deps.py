"""FastAPI dependencies (WP-D)."""

from __future__ import annotations

import logging

from fastapi import Depends, Header, HTTPException, Request, WebSocket

from synapse.config import settings

logger = logging.getLogger("synapse.api.deps")


def get_engine(request: Request):
    """The connected KnowledgeEngine, set on app.state by the lifespan.

    Tests override this via app.dependency_overrides[get_engine].
    """
    return request.app.state.engine


def require_api_key(x_synapse_key: str | None = Header(default=None)) -> None:
    """Require X-Synapse-Key header when settings.api_key is non-empty.

    When api_key is empty (the default) this dependency is a no-op, preserving
    the existing unauthenticated dev flow.
    """
    if not settings.api_key:
        return  # auth disabled
    if x_synapse_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-Synapse-Key")


async def require_ws_api_key(websocket: WebSocket) -> bool:
    """The same contract as ``require_api_key``, for the WebSocket handshake.

    /ws broadcasts every knowledge write as it happens, so leaving it open while the REST API
    is closed protects the queries and publishes the answers anyway.

    The key is accepted from the ``X-Synapse-Key`` header *or* a ``?key=`` query parameter,
    because a browser cannot set headers on a WebSocket handshake — header-only auth here
    would mean "authenticated for scripts, unreachable for the UI".

    Returns False having already rejected the socket; the caller must then return without
    calling ``accept()``. Closing before accept sends an HTTP 403 rather than opening and
    immediately dropping the connection.
    """
    if not settings.api_key:
        return True  # auth disabled — same default-open dev flow as the REST dependency
    supplied = websocket.headers.get("x-synapse-key") or websocket.query_params.get("key")
    if supplied != settings.api_key:
        logger.warning("rejected /ws handshake: invalid or missing key")
        await websocket.close(code=1008)  # 1008 = policy violation
        return False
    return True
