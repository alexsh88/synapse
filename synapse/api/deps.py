"""FastAPI dependencies (WP-D)."""

from __future__ import annotations

import logging

from fastapi import Depends, Header, HTTPException, Request

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
