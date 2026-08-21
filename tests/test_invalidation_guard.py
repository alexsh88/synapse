"""Graphiti's automatic edge-invalidation over-reaches; the write path must not believe it blindly.

Measured 2026-07-27 on the live graph: one `convention` about a service's port numbers invalidated five
valid edges, two of them about TimescaleDB. The failure is silent — no error, and the write reports
success — so these tests exist to keep the guard honest, and especially to keep anyone from
"simplifying" its two-part condition back into the bug.
"""

from __future__ import annotations

from datetime import datetime, timezone

from synapse.core.consolidation_engine import (
    carries_distinct_values,
    could_replace,
    invalidation_is_credible,
    subject_overlap,
)
from synapse.core.write_pipeline import WritePipeline

# The real pair from the incident, and the fact the write actually extracted.
TIMESCALE_FACT = "TimescaleDB is used in acme-sim."
GATEWAY_FACT = (
    "acme-sim connects to the shared ict-ib-gateway container using clientIds 17-21 for isolation."
)
# A real contradiction: same subject and relation, different value.
PORT_4003 = "The acme-data gateway is published on host port 4003."
PORT_4005 = "The acme-data gateway is published on host port 4005."


# --- the deterministic rule ----------------------------------------------------------

def test_a_value_difference_alone_does_not_justify_invalidation():
    """THE REGRESSION PIN. Both halves are required; either alone reintroduces the incident.

    `carries_distinct_values` was built for pairs a vector search already called near-identical.
    Between unrelated facts almost every token differs, so a digit-bearing one nearly always turns
    up — here `17-21` — and the value test says "yes" about two facts sharing only the word
    "acme-sim". All 5 wrongly-invalidated edges passed the value test. Only the conjunction rejects
    them, so this asserts the value test fires and the gate still refuses.
    """
    assert carries_distinct_values(TIMESCALE_FACT, GATEWAY_FACT) is True
    assert invalidation_is_credible(TIMESCALE_FACT, GATEWAY_FACT) is False


def test_same_subject_different_value_is_credible():
    assert invalidation_is_credible(PORT_4003, PORT_4005) is True


def test_a_restatement_is_not_a_contradiction():
    """Full vocabulary overlap with no differing value is a rephrasing — the merge case."""
    a = "Use ObjectProvider for optional port injections in multi-module apps."
    b = "For optional port injections in multi-module apps, use ObjectProvider."
    assert subject_overlap(a, b) == 1.0
    assert invalidation_is_credible(a, b) is False


def test_a_more_complete_restatement_does_not_retire_the_original():
    """The other 3 edges from the incident: the new fact restated them with more detail."""
    old = "Acme-Sim shares acme-data's gateway on host port 4003 with clientId 17."
    new = (
        "acme-sim connects to the shared ict-ib-gateway container using clientIds 17-21 "
        "for isolation, published on host port 4003 mapping to container port 4004."
    )
    assert invalidation_is_credible(old, new) is False


def test_overlap_divides_by_the_larger_fact():
    """A short fact that is a vocabulary subset of a long one is not the same subject.

    Dividing by the smaller side would score this 1.0 and let any sufficiently long acme-sim fact
    invalidate `TimescaleDB is used in acme-sim`.
    """
    short = "acme-sim uses TimescaleDB"
    long = (
        "acme-sim uses TimescaleDB hypertables for market data storage with a bigserial primary "
        "key including the partition column, deployed via docker compose on the trading host."
    )
    assert subject_overlap(short, long) < 0.35
    assert invalidation_is_credible(short, long) is False


def test_empty_or_unparseable_facts_never_look_credible():
    assert subject_overlap("", PORT_4003) == 0.0
    assert invalidation_is_credible("", PORT_4003) is False
    assert invalidation_is_credible(PORT_4003, "") is False


# --- the write-path guard -----------------------------------------------------------

class _Driver:
    """Fake Neo4j driver. `expired` is what the audit query finds."""

    def __init__(self, expired: list[dict] | None = None, fail: bool = False):
        self.calls: list[tuple[str, dict]] = []
        self._expired = expired or []
        self._fail = fail

    async def execute_query(self, query, **params):
        self.calls.append((query, params))
        if self._fail:
            raise RuntimeError("neo4j is down")

        class _Record(dict):
            def __getitem__(self, key):
                return dict.__getitem__(self, key)

        class _R:
            records = [_Record(r) for r in self._expired] if "expired_at" in query else []

        return _R()

    def reverts(self) -> list[dict]:
        return [p for q, p in self.calls if "invalidation_reverted_at" in q]


def _pipeline(driver):
    class _G:
        def __init__(self, d):
            self.driver = d

    return WritePipeline(graphiti=_G(driver), embedder=None, index=None, triage=None)


def _expired_row(fact: str, uuid: str = "edge-1", invalid_at: str = "2026-07-27T09:20:00Z",
                 src: str = "node-a", dst: str = "node-b", name: str = "USES"):
    return {"uuid": uuid, "fact": fact, "scope": "cluster_trading", "invalid_at": invalid_at,
            "src": src, "dst": dst, "name": name}


#: The shape of an edge that COULD replace the default _expired_row: same entity pair.
SAME_PAIR = [("node-a", "node-b", "USES")]


async def test_guard_reverts_an_unjustified_invalidation():
    driver = _Driver(expired=[_expired_row(TIMESCALE_FACT)])
    reverted = await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[GATEWAY_FACT],
        own_edge_uuids=["new-1"],
    )
    assert reverted == ["edge-1"]
    (params,) = driver.reverts()
    assert params["rows"][0]["uuid"] == "edge-1"
    # R4: the decision being undone is preserved, not erased.
    assert params["rows"][0]["from"] == "2026-07-27T09:20:00Z"
    assert "overlap" in params["rows"][0]["reason"]


async def test_guard_leaves_a_credible_invalidation_alone():
    driver = _Driver(expired=[_expired_row(PORT_4003)])
    reverted = await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[PORT_4005],
        own_edge_uuids=[],
    )
    assert reverted == []
    assert driver.reverts() == []


async def test_guard_excludes_the_writes_own_new_edges():
    """A freshly created edge is not collateral damage — it must never be fed to the audit."""
    driver = _Driver(expired=[_expired_row(TIMESCALE_FACT)])
    await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[GATEWAY_FACT],
        own_edge_uuids=["new-1", "new-2"],
    )
    audit = next(p for q, p in driver.calls if "expired_at" in q)
    assert audit["own"] == ["new-1", "new-2"]


async def test_guard_writes_timestamps_as_strings():
    """A datetime-valued custom property breaks Graphiti's later dedupe prompts for the whole
    scope, and this guard writes to edges on EVERY write — so it is the worst possible place to
    store one. Pinned because the symptom surfaces on a later, unrelated write."""
    driver = _Driver(expired=[_expired_row(TIMESCALE_FACT)])
    await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[GATEWAY_FACT],
        own_edge_uuids=[],
    )
    (params,) = driver.reverts()
    assert isinstance(params["now"], str)
    assert all(isinstance(row["from"], (str, type(None))) for row in params["rows"])


async def test_guard_does_nothing_when_no_facts_were_extracted():
    """Zero extracted facts means there is nothing to judge against, so Graphiti's decision
    stands rather than being second-guessed on no evidence."""
    driver = _Driver(expired=[_expired_row(TIMESCALE_FACT)])
    assert await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[], own_edge_uuids=[],
    ) == []
    assert driver.calls == []


async def test_guard_never_raises_when_the_driver_fails():
    """The write has already succeeded by this point and must not be undone by an audit failure."""
    driver = _Driver(expired=[_expired_row(TIMESCALE_FACT)], fail=True)
    assert await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[GATEWAY_FACT], own_edge_uuids=[],
    ) == []


# --- the structural rule (mirrors getzep/graphiti#1729) -------------------------------
# Added 2026-08-19. The lexical veto above, measured against the live graph, had reverted 11 of
# 526 retirements (2%), while this structural test applied retrospectively would have blocked
# 75.6% of the 479 whose originating write could be identified. A veto that agrees with reality
# 2% of the time is decoration.

RETIRED = ("node-a", "node-b", "USES")


def test_the_same_entity_pair_could_replace():
    assert could_replace(("node-a", "node-b", "PREFERS"), RETIRED)


def test_direction_is_not_load_bearing():
    """Extraction direction is not stable, so a restatement can legitimately arrive reversed."""
    assert could_replace(("node-b", "node-a", "USES"), RETIRED)


def test_one_shared_endpoint_plus_the_same_relation_is_a_repoint():
    assert could_replace(("node-a", "node-c", "USES"), RETIRED)
    assert could_replace(("node-c", "node-b", "USES"), RETIRED)


def test_relation_names_fold_across_case_and_separators():
    assert could_replace(("node-a", "node-c", "uses"), RETIRED)
    assert could_replace(("node-a", "node-c", "US_ES"), ("node-a", "node-b", "us-es"))


def test_a_shared_endpoint_with_a_different_relation_could_not_replace():
    """The core of the bug: merely MENTIONING an entity is not a claim about this relationship."""
    assert not could_replace(("node-a", "node-c", "OWNS"), RETIRED)


def test_two_unrelated_entities_could_not_replace():
    assert not could_replace(("node-x", "node-y", "USES"), RETIRED)


async def test_guard_reverts_when_no_new_edge_could_structurally_replace():
    """Even when the prose passes the lexical test, an unrelated relationship cannot supersede."""
    driver = _Driver(expired=[_expired_row(PORT_4003)])
    reverted = await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[PORT_4005],
        own_edge_uuids=["new-1"],
        new_edges=[("node-x", "node-y", "MENTIONS")],
    )
    assert reverted == ["edge-1"]
    (params,) = driver.reverts()
    assert "could have replaced" in params["rows"][0]["reason"]


async def test_guard_upholds_when_both_halves_agree():
    driver = _Driver(expired=[_expired_row(PORT_4003)])
    assert await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[PORT_4005], own_edge_uuids=["new-1"], new_edges=SAME_PAIR,
    ) == []


async def test_a_structurally_plausible_edge_still_needs_the_lexical_half():
    """Both halves are required. Same entity pair, but the new fact is about something else."""
    driver = _Driver(expired=[_expired_row(TIMESCALE_FACT)])
    assert await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[GATEWAY_FACT], own_edge_uuids=["new-1"], new_edges=SAME_PAIR,
    ) == ["edge-1"]


async def test_the_structural_half_abstains_when_the_retired_shape_is_unreadable():
    """Unknown must not mean "no replacement exists" — that would revert everything the moment a
    query changed shape. Abstaining leaves the decision to the lexical test, which keeps facts."""
    row = _expired_row(PORT_4003)
    row["src"] = None
    driver = _Driver(expired=[row])
    assert await _pipeline(driver)._revert_unjustified_invalidations(
        since=datetime(2026, 7, 27, 9, 20, tzinfo=timezone.utc),
        new_facts=[PORT_4005], own_edge_uuids=["new-1"],
        new_edges=[("node-x", "node-y", "MENTIONS")],
    ) == []
