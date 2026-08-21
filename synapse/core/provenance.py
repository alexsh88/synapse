"""Write provenance — who taught Synapse this, with what, and when (roadmap item 13).

Research §5.2. Before this, ``source_description`` carried exactly four values across the whole
corpus (``synapse:seed`` 112, ``synapse:agent`` 93, ``synapse:capture`` 90, ``synapse:connect`` 15).
There was no writer identity at all, which the multi-agent memory literature calls **provenance
collapse**. Two concrete costs:

* You cannot ask *"what did this bad session teach us?"* and roll it back. Every edge carries an
  ``episodes`` list (verified: 3,030 of 3,030), so once episodes carry a session id the blast radius
  of a bad write becomes a single query.
* You cannot weight ranking by source reliability, which is the prerequisite for the agent tier and
  reputation-weighted retrieval (roadmap item 19).

This is the one item on the roadmap that is **cheap now and impossible retroactively**: knowledge
already stored can never learn where it came from.

Properties are written with a ``prov_`` prefix. Graphiti owns the ``Episodic`` node's schema
(``content``, ``source``, ``source_description``, ``valid_at``, ``entity_edges`` …) and adds fields
across versions, so an unprefixed ``model`` or ``host`` risks a future collision on a node type we
do not control.
"""

from __future__ import annotations

import os
import socket

from pydantic import BaseModel, Field

# Env vars an agent/host can set once so every write it makes is attributed without threading
# arguments through each call site.
_ENV_AGENT = "SYNAPSE_AGENT"
_ENV_SESSION = "SYNAPSE_SESSION_ID"
_ENV_MODEL = "SYNAPSE_MODEL"
# Explicit host name. Needed because the API runs in a container: verified live, an unaided
# ``socket.gethostname()`` returned "9c03e1a73872" — the container id, which is ephemeral and tells
# you nothing about which machine actually wrote the knowledge. Set SYNAPSE_HOST in the compose file
# to attribute writes to the real host.
_ENV_HOST = "SYNAPSE_HOST"
# Hostname is useful once more than one machine writes, but it is also the most identifying field
# here, so it is the one thing that can be switched off.
_ENV_HOST_OPTOUT = "SYNAPSE_PROV_NO_HOST"

_PREFIX = "prov_"


class Provenance(BaseModel):
    """Who/what produced a write. Every field is optional — partial attribution beats none."""

    agent: str | None = Field(None, description="The writer: an agent name or role (claude-code, capture).")
    model: str | None = Field(None, description="Model that authored the content, e.g. claude-opus-5.")
    session_id: str | None = Field(None, description="Session the write came from — the rollback key.")
    host: str | None = Field(None, description="Machine that performed the write.")

    def as_props(self) -> dict[str, str]:
        """Neo4j properties for the stored episode. Omits empty fields entirely.

        Omitting rather than writing nulls keeps ``count(e.prov_session_id)`` a usable coverage
        measure, the same way the corpus scan reads ``count(e.content_hash)``.
        """
        return {
            f"{_PREFIX}{key}": value.strip()
            for key, value in self.model_dump().items()
            if isinstance(value, str) and value.strip()
        }

    def is_empty(self) -> bool:
        return not self.as_props()


def _hostname() -> str | None:
    """The writing host: opt-out, then SYNAPSE_HOST, then the OS hostname."""
    if os.environ.get(_ENV_HOST_OPTOUT, "").strip().lower() in ("1", "true", "yes"):
        return None
    configured = os.environ.get(_ENV_HOST, "").strip()
    if configured:
        return configured
    try:
        return socket.gethostname() or None
    except Exception:  # noqa: BLE001 — attribution must never break a write
        return None


def resolve(
    explicit: Provenance | None = None,
    *,
    agent: str | None = None,
    session_id: str | None = None,
    model: str | None = None,
) -> Provenance:
    """Build provenance from explicit values, then keyword hints, then the environment.

    Precedence is explicit > keyword > env, so a caller that genuinely knows (an API request, a
    hook that received a real session id) always wins over an ambient default.
    """
    base = explicit.model_dump() if explicit else {}

    def pick(field: str, hint: str | None, env_var: str) -> str | None:
        for candidate in (base.get(field), hint, os.environ.get(env_var)):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    return Provenance(
        agent=pick("agent", agent, _ENV_AGENT),
        model=pick("model", model, _ENV_MODEL),
        session_id=pick("session_id", session_id, _ENV_SESSION),
        host=base.get("host") or _hostname(),
    )
