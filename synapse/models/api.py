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
    """Response model for POST /knowledge (remember). Mirrors WriteResult's public surface.

    ``extra="allow"`` is kept, but do NOT rely on it to forward new fields. It only accepts extras
    when validating a *mapping*; FastAPI validates the returned WriteResult **object**, and pydantic
    does not harvest unknown attributes off an object as extras. So every field a caller needs must
    be declared here explicitly.

    That was found the hard way twice: `degraded`/`facts_extracted` had to be added by hand, and
    then the global-write gate's `scope_redirected_from` and the `reason` text both came back null
    from a live request even though the pipeline had set them — the redirect happened but the caller
    could not see why. Adding a field to WriteResult means adding it here too.
    """

    model_config = ConfigDict(extra="allow")

    outcome: str
    reason: str = ""
    knowledge_type: str | None = None
    scope: str | None = None
    episode_uuid: str | None = None
    entities: list[str] = []
    facts: list[str] = []
    duplicate_of: str | None = None
    contradicts: str | None = None
    # WP-B diagnostics (WriteResult gained these; must not be stripped)
    degraded: bool = False
    facts_extracted: int = 0
    # Credential kinds stripped before storage (research §5.1) — kinds only, never values.
    redactions: list[str] = []
    # Set when the global-write gate refiled the write (research §5.3, roadmap item 15).
    scope_redirected_from: str | None = None
    # Instruction-shaped content caught heading for a broadcast scope. Kinds only, never the
    # payload — a caller whose write was contained or refused has to be able to see why.
    injection_kinds: list[str] = []


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
    # The write OUTCOME for the Project entity ("stored" / "duplicate" / ...), not the entity
    # itself. Typed `dict` here until 2026-07-30, which made every successful connect fail
    # response validation. The UI renders it as text and its TS type says `entity?: string`.
    entity: str | None = None


# --- EventBus queue size — single source of truth lives in events.py ---
# Import and re-export so any existing code that imports QUEUE_MAXSIZE from
# here continues to work, while the real definition is in the module that uses it.
from synapse.api.events import _QUEUE_MAXSIZE as QUEUE_MAXSIZE  # noqa: E402
