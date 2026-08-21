"""Procedural memory — ordered runbooks (roadmap item 18).

The invariant under test throughout: **order is the payload**. A runbook whose steps have been
reordered, deduplicated or summarized is not a degraded runbook, it is a wrong one — which is why
these tests are pickier about mutation than the semantic types' tests are.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from synapse.core.runbooks import (
    MAX_STEP_CHARS,
    MAX_STEPS,
    RunbookRecord,
    RunbookStore,
    normalize_steps,
    runbook_prose,
)
from synapse.core.schema import ENTITY_TYPES

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


# --- the type is deliberately kept away from the extractor -------------------


def test_runbook_is_not_an_extractable_entity_type():
    """The whole design rests on this: extraction is what destroys step order.

    Adding `Runbook` to ENTITY_TYPES would let the LLM mint runbooks from prose — producing
    runbooks with no steps, i.e. the exact broken state measured on the live graph, wearing a
    label that claims otherwise.
    """
    assert "Runbook" not in ENTITY_TYPES


# --- step normalization ------------------------------------------------------


def test_normalize_preserves_order():
    steps = ["stop the gateway", "flush redis", "start the gateway"]
    assert normalize_steps(steps) == steps


def test_normalize_keeps_duplicate_steps():
    # "Restart the gateway" legitimately appears twice in a procedure. Deduplicating would
    # silently change what the operator is told to do — the one mutation this type must not make.
    steps = ["restart the gateway", "wait 30s", "restart the gateway"]
    assert normalize_steps(steps) == steps


def test_normalize_collapses_whitespace_but_not_content():
    assert normalize_steps(["  run   the   migration \n"]) == ["run the migration"]


def test_normalize_drops_blank_steps():
    assert normalize_steps(["a", "   ", "b"]) == ["a", "b"]


def test_normalize_rejects_an_empty_runbook():
    with pytest.raises(ValueError, match="at least one step"):
        normalize_steps([])
    with pytest.raises(ValueError, match="at least one step"):
        normalize_steps(["", "  "])


def test_normalize_rejects_absurd_length():
    with pytest.raises(ValueError, match="cap is"):
        normalize_steps([f"step {i}" for i in range(MAX_STEPS + 1)])


def test_normalize_rejects_a_step_that_is_really_a_procedure():
    with pytest.raises(ValueError, match="split"):
        normalize_steps(["x" * (MAX_STEP_CHARS + 1)])


# --- staleness ---------------------------------------------------------------


def _record(**kw) -> RunbookRecord:
    base = {"uuid": "u", "name": "deploy", "scope": "project_x", "steps": ["a"]}
    return RunbookRecord(**{**base, **kw})


def test_never_verified_counts_as_stale():
    # A procedure nobody has run is exactly as trustworthy as one last run a year ago.
    assert _record(verified_at=None).is_stale(now=NOW) is True


def test_recently_verified_is_not_stale():
    assert _record(verified_at=NOW - timedelta(days=3)).is_stale(now=NOW) is False


def test_verification_expires():
    assert _record(verified_at=NOW - timedelta(days=200)).is_stale(now=NOW) is True


def test_as_lines_numbers_the_steps():
    assert _record(steps=["a", "b"]).as_lines() == ["1. a", "2. b"]


def test_prose_contains_every_step_for_the_search_index():
    text = runbook_prose("deploy acme-api", ["build", "smoke test", "promote"],
                         "ship a release", "gateway is up")
    for step in ("build", "smoke test", "promote"):
        assert step in text
    assert "ship a release" in text
    assert "gateway is up" in text


# --- the store ---------------------------------------------------------------


class FakeResult:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    """Records queries and replays canned rows, so store logic is testable without Neo4j."""

    def __init__(self):
        self.node = None          # the "stored" runbook row
        self.queries = []

    async def execute_query(self, query, **params):
        self.queries.append((query, params))
        if "RETURN n.uuid AS uuid, n.steps AS steps" in query:      # the adopt-or-create probe
            return FakeResult([{"uuid": self.node["uuid"], "steps": self.node["steps"]}]
                              if self.node else [])
        if query.strip().startswith("CREATE"):
            self.node = {
                "uuid": params["uuid"], "name": params["name"], "scope": params["scope"],
                "steps": params["steps"], "purpose": params["purpose"],
                "prerequisites": params["prerequisites"], "verified_at": params["verified_at"],
                "created_at": params["now"], "updated_at": params["now_iso"],
                "previous_steps": [], "superseded_at": None,
            }
            return FakeResult([{"uuid": params["uuid"]}])
        if query.strip().startswith("MATCH (n:Entity {uuid:"):       # the update
            self.node.update(
                steps=params["steps"], purpose=params["purpose"],
                prerequisites=params["prerequisites"], verified_at=params["verified_at"],
                updated_at=params["now_iso"], previous_steps=params["previous_steps"],
                superseded_at=params["superseded_at"],
            )
            return FakeResult([{"uuid": self.node["uuid"]}])
        if "MATCH (n:Runbook)" in query:                             # the reads
            if self.node is None:
                return FakeResult([])
            if "n.name = $name" in query and params.get("name") != self.node["name"]:
                return FakeResult([])
            if self.node["scope"] not in params["scopes"]:
                return FakeResult([])
            return FakeResult([dict(self.node)])
        return FakeResult([])


class FakeGraph:
    def __init__(self):
        self.driver = FakeDriver()


async def test_upsert_creates_a_runbook_with_its_steps():
    graph = FakeGraph()
    rec = await RunbookStore(graph).upsert(
        name="deploy acme-api", scope="project_acme-api",
        steps=["build", "smoke test", "promote"], purpose="ship a release", now=NOW,
    )
    assert rec.steps == ["build", "smoke test", "promote"]
    assert rec.scope == "project_acme-api"
    assert rec.purpose == "ship a release"


async def test_upsert_adopts_an_existing_node_rather_than_duplicating():
    # The prose episode is written first, so extraction has usually already made a node with this
    # name. Adopting it keeps the procedure connected to everything the extractor linked.
    graph = FakeGraph()
    store = RunbookStore(graph)
    first = await store.upsert(name="deploy", scope="project_x", steps=["a"], now=NOW)
    second = await store.upsert(name="deploy", scope="project_x", steps=["a", "b"], now=NOW)
    assert first.uuid == second.uuid
    creates = [q for q, _ in graph.driver.queries if q.strip().startswith("CREATE")]
    assert len(creates) == 1


async def test_replacing_steps_supersedes_rather_than_deletes():
    # R4: knowledge supersedes, it is not deleted. "What did the deploy sequence look like in
    # May" must stay answerable for procedures too, not just facts.
    graph = FakeGraph()
    store = RunbookStore(graph)
    await store.upsert(name="deploy", scope="project_x", steps=["old-1", "old-2"], now=NOW)
    updated = await store.upsert(name="deploy", scope="project_x", steps=["new-1"], now=NOW)
    assert updated.steps == ["new-1"]
    assert updated.previous_steps == ["old-1", "old-2"]
    assert updated.superseded_at == NOW


async def test_reverifying_unchanged_steps_does_not_push_a_duplicate_into_history():
    graph = FakeGraph()
    store = RunbookStore(graph)
    await store.upsert(name="deploy", scope="project_x", steps=["a", "b"], now=NOW)
    again = await store.upsert(
        name="deploy", scope="project_x", steps=["a", "b"], verified_at=NOW, now=NOW,
    )
    assert again.previous_steps == []
    assert again.superseded_at is None
    assert again.verified_at == NOW


async def test_upsert_rejects_a_nameless_runbook():
    with pytest.raises(ValueError, match="needs a name"):
        await RunbookStore(FakeGraph()).upsert(name="   ", scope="project_x", steps=["a"])


async def test_upsert_rejects_a_stepless_runbook():
    with pytest.raises(ValueError, match="at least one step"):
        await RunbookStore(FakeGraph()).upsert(name="deploy", scope="project_x", steps=[])


async def test_list_is_scoped():
    graph = FakeGraph()
    store = RunbookStore(graph)
    await store.upsert(name="deploy", scope="project_acme-api", steps=["a"], now=NOW)
    assert len(await store.list_for_scopes(["global", "project_acme-api"])) == 1
    assert await store.list_for_scopes(["project_acme-docs"]) == []


async def test_custom_properties_are_json_serializable():
    """A datetime-valued custom property on an Entity node breaks the NEXT unrelated write.

    Graphiti serializes an existing node's non-core properties to JSON for its node-dedupe
    prompt, and `json.dumps` cannot encode a `neo4j.time.DateTime`. The first live runbook wrote
    fine and then poisoned every subsequent `add_episode` into `project_synapse` with
    `TypeError: Object of type DateTime is not JSON serializable`. Timestamps are therefore
    stored as ISO strings — this test is the guard, because the symptom appears somewhere else
    entirely and would be very hard to trace back here.
    """
    graph = FakeGraph()
    store = RunbookStore(graph)
    await store.upsert(name="deploy", scope="project_x", steps=["a"], verified_at=NOW, now=NOW)
    await store.upsert(name="deploy", scope="project_x", steps=["b"], verified_at=NOW, now=NOW)

    custom = ("steps", "purpose", "prerequisites", "verified_at", "updated_at",
              "previous_steps", "superseded_at")
    for query, params in graph.driver.queries:
        if query.strip().startswith(("CREATE", "MATCH (n:Entity {uuid:")):
            payload = {k: v for k, v in params.items() if k in custom}
            json.dumps(payload)   # raises TypeError if any datetime slipped through
            assert not isinstance(params.get("verified_at"), datetime)
            assert not isinstance(params.get("superseded_at"), datetime)


async def test_timestamps_survive_the_iso_round_trip():
    graph = FakeGraph()
    rec = await RunbookStore(graph).upsert(
        name="deploy", scope="project_x", steps=["a"], verified_at=NOW, now=NOW,
    )
    assert rec.verified_at == NOW
    assert rec.is_stale(now=NOW) is False


async def test_store_without_a_driver_reads_empty_and_refuses_to_write():
    class NoDriver:
        driver = None

    store = RunbookStore(NoDriver())
    assert await store.list_for_scopes(["global"]) == []
    with pytest.raises(RuntimeError, match="driver"):
        await store.upsert(name="deploy", scope="global", steps=["a"])
