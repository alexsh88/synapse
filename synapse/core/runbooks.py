"""Procedural memory — ordered, executable runbooks (research §3, roadmap item 18).

Every other knowledge type in Synapse is written by handing free text to Graphiti and letting an
LLM extract entities from it. Runbooks cannot work that way, and this module exists because of a
measurement rather than a preference: on the live graph, *not one step of any procedure the graph
references was actually in the graph*. What survived extraction was a node **name** with an arrow
chain in it, and two checklists reduced to the numbers "10-item" and "13-point".

So the ordered steps are written **deterministically, as a node property**, and never routed
through extraction. The prose episode still goes through the normal pipeline — that is what makes
a runbook findable by `recall()`, which searches fact edges — but the authoritative ordering lives
on the node where no model can rewrite it.

Design notes
------------
* **Identity is (name, scope).** A runbook is a named procedure within a scope; re-writing it with
  the same name is an *update*, not a second runbook. Anything else and the graph accumulates
  three subtly different deploy sequences with no way to tell which one is current.
* **Adopt, don't duplicate.** If extraction already created an entity node with this exact name in
  this scope, the runbook attaches to it rather than creating a rival. The prose episode is
  written first precisely so this adoption is the normal case, which keeps the procedure connected
  to everything the extractor linked it to.
* **Supersede, never silently rewrite** (R4). Updating a runbook keeps the previous steps in
  ``previous_steps`` with ``superseded_at``, so "what did the deploy sequence look like in May" is
  answerable — the same guarantee the temporal model gives facts.
"""

from __future__ import annotations

import logging
import uuid as uuid_lib
from datetime import datetime, timezone

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# A runbook with no steps is the exact failure this type exists to prevent — it would be a
# Convention wearing a label that promises more than it holds.
MIN_STEPS = 1
MAX_STEPS = 60
MAX_STEP_CHARS = 600


class RunbookRecord(BaseModel):
    """A stored procedure. ``steps`` is ordered and the order is the payload."""

    uuid: str
    name: str
    scope: str
    steps: list[str]
    purpose: str | None = None
    prerequisites: str | None = None
    verified_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    previous_steps: list[str] = Field(default_factory=list)
    superseded_at: datetime | None = None

    def as_lines(self) -> list[str]:
        """The runbook as numbered text — for briefs and for the searchable prose episode."""
        return [f"{i}. {step}" for i, step in enumerate(self.steps, start=1)]

    def is_stale(self, *, now: datetime | None = None, max_age_days: float = 90.0) -> bool:
        """True when the steps have not been confirmed to work recently enough.

        Never verified counts as stale. A procedure nobody has run is exactly as trustworthy as one
        last run a year ago, and pretending otherwise is how a brief recommends a broken sequence.
        """
        if self.verified_at is None:
            return True
        now = now or datetime.now(timezone.utc)
        return (now - self.verified_at).total_seconds() / 86400.0 > max_age_days


def normalize_steps(steps: list[str]) -> list[str]:
    """Trim and validate step text, preserving order and duplicates.

    Duplicates are **kept on purpose**. "Restart the gateway" legitimately appears twice in a
    procedure, and deduplicating it would silently change what the operator is told to do — the
    one mutation a procedural type must never make.
    """
    cleaned = [" ".join(s.split()) for s in steps]
    cleaned = [s for s in cleaned if s]
    if len(cleaned) < MIN_STEPS:
        raise ValueError(
            "a runbook needs at least one step — a step-less runbook is the exact gap this type "
            "exists to close (see synapse.core.schema.Runbook)"
        )
    if len(cleaned) > MAX_STEPS:
        raise ValueError(f"runbook has {len(cleaned)} steps; the cap is {MAX_STEPS}")
    too_long = [s for s in cleaned if len(s) > MAX_STEP_CHARS]
    if too_long:
        raise ValueError(
            f"{len(too_long)} step(s) exceed {MAX_STEP_CHARS} chars — a step that long is a "
            "procedure of its own and should be split"
        )
    return cleaned


def runbook_prose(name: str, steps: list[str], purpose: str | None,
                  prerequisites: str | None) -> str:
    """The searchable text form written through the normal episode pipeline.

    This is what makes a runbook reachable by `recall()` (which searches fact edges, not nodes).
    Extraction will mangle the ordering here and that is fine — this copy is the index, not the
    source of truth.
    """
    parts = [f"Runbook: {name}."]
    if purpose:
        parts.append(f"Purpose: {purpose}.")
    if prerequisites:
        parts.append(f"Prerequisites: {prerequisites}.")
    parts.append("Steps: " + " ".join(f"{i}. {s}" for i, s in enumerate(steps, start=1)))
    return " ".join(parts)


_UPSERT = """
MATCH (n:Entity {name: $name, group_id: $scope})
WHERE coalesce(n.archived, false) = false
RETURN n.uuid AS uuid, n.steps AS steps
LIMIT 1
"""

_WRITE = """
MATCH (n:Entity {uuid: $uuid})
SET n:Runbook,
    n.steps = $steps,
    n.purpose = $purpose,
    n.prerequisites = $prerequisites,
    n.verified_at = $verified_at,
    n.updated_at = $now_iso,
    n.summary = coalesce($purpose, n.summary),
    n.previous_steps = $previous_steps,
    n.superseded_at = $superseded_at
RETURN n.uuid AS uuid
"""

_CREATE = """
CREATE (n:Entity:Runbook {
    uuid: $uuid, name: $name, group_id: $scope, steps: $steps,
    purpose: $purpose, prerequisites: $prerequisites, verified_at: $verified_at,
    summary: $purpose, created_at: $now, updated_at: $now_iso,
    previous_steps: [], superseded_at: null
})
RETURN n.uuid AS uuid
"""

_READ = """
MATCH (n:Runbook)
WHERE n.group_id IN $scopes
  AND coalesce(n.archived, false) = false
  AND n.invalid_at IS NULL
RETURN n.uuid AS uuid, n.name AS name, n.group_id AS scope, n.steps AS steps,
       n.purpose AS purpose, n.prerequisites AS prerequisites,
       n.verified_at AS verified_at, n.created_at AS created_at, n.updated_at AS updated_at,
       coalesce(n.previous_steps, []) AS previous_steps, n.superseded_at AS superseded_at
ORDER BY coalesce(n.updated_at, n.created_at) DESC
LIMIT $limit
"""

_READ_ONE = _READ.replace(
    "WHERE n.group_id IN $scopes", "WHERE n.group_id IN $scopes AND n.name = $name"
)


def _iso(value: datetime | None) -> str | None:
    """Timestamps are stored as ISO-8601 STRINGS, not Neo4j DateTime. This is not cosmetic.

    Graphiti serializes an existing node's non-core properties to JSON when it builds its
    node-dedupe prompt, and `json.dumps` cannot encode a `neo4j.time.DateTime`. Writing a
    datetime-valued custom property onto an `Entity` node therefore breaks **every subsequent
    `add_episode` into that scope** with `TypeError: Object of type DateTime is not JSON
    serializable` — not the runbook write, which succeeds, but the next unrelated one.

    Found the hard way: the first live runbook stored fine, then poisoned normal writes to
    `project_synapse`. `created_at` stays a real DateTime because it is a core Graphiti field and
    is excluded from that serialization; only our custom properties need to be strings.
    """
    return value.isoformat() if value is not None else None


def _native(value):
    """Read side of :func:`_iso` — ISO string or neo4j.time.DateTime -> datetime."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return value.to_native() if hasattr(value, "to_native") else value


class RunbookStore:
    """Reads and writes `Runbook` nodes directly. No extractor, no embedder, no LLM.

    Takes anything exposing ``.driver`` (a Graphiti instance or
    :class:`synapse.db.neo4j_client.DirectGraph`), matching the convention the maintenance
    scripts already use.

    **Known consequence of staying embedder-free:** a node this class CREATES (rather than adopts
    from extraction) has no ``name_embedding``, so Graphiti's node-similarity search cannot see it
    until one is backfilled by ``scripts/reembed_corpus.py``, which selects exactly on "vector
    missing or wrong dimension". Adoption is the common path — the prose episode is written first
    — so this only bites when the extractor happened to name its node differently. Accepted rather
    than fixed here, because dragging an embedder into the one write path that must never depend
    on a model would trade a small retrieval gap for the very coupling this type exists to avoid.
    Runbooks remain findable meanwhile via `brief()`, `runbooks()` and the prose episode's facts.
    """

    def __init__(self, graph) -> None:
        self._graph = graph

    def _driver(self):
        return getattr(self._graph, "driver", None)

    async def upsert(
        self,
        *,
        name: str,
        scope: str,
        steps: list[str],
        purpose: str | None = None,
        prerequisites: str | None = None,
        verified_at: datetime | None = None,
        now: datetime | None = None,
    ) -> RunbookRecord:
        """Create or update the runbook named *name* in *scope*.

        Identity is (name, scope). When an entity node with that name already exists — usually
        because the prose episode was written first and extraction created one — this **adopts**
        it, adding the label and the steps rather than creating a rival node.

        Replacing existing steps preserves them in ``previous_steps`` with ``superseded_at`` (R4:
        knowledge supersedes, it is not deleted).
        """
        driver = self._driver()
        if driver is None:
            raise RuntimeError("RunbookStore requires a Neo4j driver")

        name = " ".join(name.split())
        if not name:
            raise ValueError("a runbook needs a name — identity is (name, scope)")
        steps = normalize_steps(steps)
        now = now or datetime.now(timezone.utc)

        existing = (await driver.execute_query(_UPSERT, name=name, scope=scope)).records
        params = {
            "name": name, "scope": scope, "steps": steps, "purpose": purpose,
            "prerequisites": prerequisites, "verified_at": _iso(verified_at),
            "now_iso": _iso(now), "now": now,
        }

        if existing:
            node_uuid = existing[0]["uuid"]
            prior = list(existing[0]["steps"] or [])
            # Only supersede when the steps actually changed; re-verifying an unchanged runbook
            # must not push a duplicate copy into history.
            changed = prior and prior != steps
            await driver.execute_query(
                _WRITE, uuid=node_uuid,
                previous_steps=prior if changed else [],
                superseded_at=_iso(now) if changed else None,
                **params,
            )
            if changed:
                logger.info(
                    "runbook %r in %s superseded: %d steps -> %d", name, scope, len(prior),
                    len(steps),
                )
            return await self._read_back(scope, name)

        node_uuid = str(uuid_lib.uuid4())
        await driver.execute_query(_CREATE, uuid=node_uuid, **params)
        logger.info("runbook %r created in %s with %d steps", name, scope, len(steps))
        return await self._read_back(scope, name)

    async def _read_back(self, scope: str, name: str) -> RunbookRecord:
        """Re-read a runbook we just wrote. Absence here is not "not found" — it means the
        write did not land, so say that rather than returning None into a non-Optional contract."""
        record = await self._one(scope, name)
        if record is None:
            raise RuntimeError(
                f"runbook {name!r} in {scope} vanished immediately after its write"
            )
        return record

    async def get(self, name: str, scopes: list[str]) -> RunbookRecord | None:
        return await self._one(scopes, name)

    async def list_for_scopes(self, scopes: list[str], limit: int = 20) -> list[RunbookRecord]:
        driver = self._driver()
        if driver is None:
            return []
        result = await driver.execute_query(_READ, scopes=scopes, limit=limit)
        return [self._record(r) for r in result.records]

    async def _one(self, scopes, name: str) -> RunbookRecord | None:
        driver = self._driver()
        if driver is None:
            return None
        scopes = [scopes] if isinstance(scopes, str) else scopes
        result = await driver.execute_query(_READ_ONE, scopes=scopes, name=name, limit=1)
        records = result.records
        return self._record(records[0]) if records else None

    @staticmethod
    def _record(row) -> RunbookRecord:
        return RunbookRecord(
            uuid=row["uuid"], name=row["name"], scope=row["scope"],
            steps=list(row["steps"] or []), purpose=row["purpose"],
            prerequisites=row["prerequisites"],
            verified_at=_native(row["verified_at"]),
            created_at=_native(row["created_at"]),
            updated_at=_native(row["updated_at"]),
            previous_steps=list(row["previous_steps"] or []),
            superseded_at=_native(row["superseded_at"]),
        )
