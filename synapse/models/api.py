"""Thin response models for the Synapse API (WP-D).

These live here (not in core/) so routes can reference them without importing
core business-logic modules.  Where a core type is already importable and fits
the response shape exactly, we re-export it from here for a single import site.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# --- Health ---

class HealthResponse(BaseModel):
    status: str


# --- Knowledge write ---

class RememberResponse(BaseModel):
    """Response model for POST /knowledge (remember).

    Mirrors WriteResult's public surface.  ``extra="allow"`` so future
    diagnostic fields added to WriteResult are forwarded to callers rather than
    silently stripped — avoids a stripping surprise when the write pipeline adds
    fields mid-flight (WP-D spec item 1).
    """

    model_config = ConfigDict(extra="allow")

    outcome: str
    knowledge_type: str | None = None
    scope: str | None = None
    episode_uuid: str | None = None
    facts: list[str] = []
    # WP-B diagnostics (WriteResult gained these; must not be stripped)
    degraded: bool = False
    facts_extracted: int = 0


class UpdateResponse(BaseModel):
    success: bool


class ForgetResponse(BaseModel):
    success: bool


# --- Capture ---

class CaptureAccepted(BaseModel):
    accepted: bool
    reason: str | None = None


class CaptureCountResponse(BaseModel):
    count: int


# --- Projects connect job ---

class ConnectJobResponse(BaseModel):
    job_id: str
    project: str
    state: str
    done: int
    total: int
    stored: int
    error: str | None = None
    actions: list[str] = []
    entity: dict | None = None


# --- EventBus queue size — single source of truth lives in events.py ---
# Import and re-export so any existing code that imports QUEUE_MAXSIZE from
# here continues to work, while the real definition is in the module that uses it.
from synapse.api.events import _QUEUE_MAXSIZE as QUEUE_MAXSIZE  # noqa: E402
