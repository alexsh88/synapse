"""Consolidation engine — the nightly propose-only night shift (research Wave 2).

Fake driver + fake curation engine, so proposal generation, idempotency, target-scope
computation and the apply paths are all tested without Neo4j.

The invariants that matter most here are SAFETY ones: propose() must never mutate knowledge,
a dismissed proposal must never come back, and applying a promotion must never invent the
statement it stores.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from synapse.core.consolidation_engine import (
    ConsolidationEngine,
    ProposalKind,
    ProposalStatus,
)

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


# --- fakes -------------------------------------------------------------------


class _Result:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    """Records every query and serves canned results keyed by a marker substring."""

    def __init__(self, promotion_rows=None, existing_keys=None, proposals=None):
        self.queries: list[tuple[str, dict]] = []
        self.promotion_rows = promotion_rows or []
        self.existing_keys = set(existing_keys or [])   # keys that already exist => not "created"
        self.proposals: dict[str, dict] = proposals or {}
        self.status_writes: list[tuple[str, str]] = []

    async def execute_query(self, query, **params):
        self.queries.append((query, params))
        if "MERGE (p:CurationProposal" in query:
            key = params["key"]
            created = key not in self.existing_keys
            self.existing_keys.add(key)
            uuid = f"prop-{len(self.existing_keys)}"
            if created:
                self.proposals[uuid] = {
                    "uuid": uuid, "key": key, "kind": params["kind"],
                    "status": ProposalStatus.OPEN.value, "rationale": params["rationale"],
                    "created_at": params["now"], **params["props"],
                }
            return _Result([{"uuid": uuid, "status": ProposalStatus.OPEN.value, "created": created}])
        if "WITH n.name AS name" in query:
            return _Result(self.promotion_rows)
        if "RETURN p AS p ORDER BY" in query:
            rows = [p for p in self.proposals.values()
                    if params.get("status") in (None, p["status"])
                    and params.get("kind") in (None, p["kind"])]
            return _Result([{"p": r} for r in rows])
        if "RETURN p AS p" in query:
            found = self.proposals.get(params["uuid"])
            return _Result([{"p": found}] if found else [])
        if "SET p.status" in query:
            self.status_writes.append((params["uuid"], params["status"]))
            if params["uuid"] in self.proposals:
                self.proposals[params["uuid"]]["status"] = params["status"]
            return _Result([])
        return _Result([])


class FakeGraphiti:
    def __init__(self, driver):
        self.driver = driver


class _Ref:
    def __init__(self, uuid, fact):
        self.uuid = uuid
        self.fact = fact


class _Cluster:
    def __init__(self, scope, canonical, duplicates, sim):
        self.scope = scope
        self.canonical = canonical
        self.duplicates = duplicates
        self.max_similarity = sim


class _ApplyResult:
    def __init__(self, ok=True, detail="superseded", backup_path="backups/x.json"):
        self.ok = ok
        self.detail = detail
        self.backup_path = backup_path


class _Pair:
    def __init__(self, scope, a, b, similarity):
        self.scope = scope
        self.a = a
        self.b = b
        self.similarity = similarity


class FakeCuration:
    def __init__(self, clusters=None, merge_ok=True, review_pairs=None):
        self.clusters = clusters or []
        self.merged: list[tuple[str, str]] = []
        self.merge_ok = merge_ok
        self.review_pairs = review_pairs or []

    async def find_duplicates(self, scopes=None):
        return self.clusters

    async def find_review_pairs(self, scopes=None):
        return self.review_pairs

    async def merge_duplicate(self, canonical, duplicate):
        self.merged.append((canonical, duplicate))
        return _ApplyResult(ok=self.merge_ok, detail="superseded" if self.merge_ok else "not found")


def _engine(*, driver=None, curation=None, cluster_resolver=None, remember=None):
    driver = driver or FakeDriver()
    return ConsolidationEngine(
        FakeGraphiti(driver), curation or FakeCuration(),
        cluster_resolver=cluster_resolver, remember=remember, now=NOW,
    ), driver


# --- merge proposals ---------------------------------------------------------


async def test_proposes_one_merge_per_duplicate_in_a_cluster():
    # Genuine rephrasings — the safety gate only lets same-content-word pairs through.
    curation = FakeCuration(clusters=[
        _Cluster("project_x", _Ref("keep", "BigDecimal is the required type for money"),
                 [_Ref("dup1", "BigDecimal is required as the type for money"),
                  _Ref("dup2", "The required type for money is BigDecimal")], 0.9812),
    ])
    engine, driver = _engine(curation=curation)
    run = await engine.propose()
    assert run.merges_proposed == 2
    keys = {q[1]["key"] for q in driver.queries if "MERGE (p:CurationProposal" in q[0]}
    assert keys == {"merge:keep:dup1", "merge:keep:dup2"}


async def test_merge_proposal_records_similarity_and_both_facts():
    curation = FakeCuration(clusters=[
        _Cluster("project_x", _Ref("keep", "the pipeline deduplicates before storing"),
                 [_Ref("dup1", "before storing, the pipeline deduplicates")], 0.99),
    ])
    engine, driver = _engine(curation=curation)
    await engine.propose()
    props = next(q[1]["props"] for q in driver.queries if "MERGE (p:CurationProposal" in q[0])
    assert props["similarity"] == pytest.approx(0.99)
    assert props["canonical_fact"] == "the pipeline deduplicates before storing"
    assert props["duplicate_fact"] == "before storing, the pipeline deduplicates"


async def test_merge_proposals_are_capped_and_the_cap_is_logged_not_silent(caplog):
    clusters = [
        _Cluster("project_x", _Ref(f"c{i}", "f"), [_Ref(f"d{i}", "f")], 0.98) for i in range(10)
    ]
    engine, _ = _engine(curation=FakeCuration(clusters=clusters))
    with caplog.at_level("INFO"):
        run = await engine.propose(max_merges=3)
    assert run.merges_proposed == 3
    assert any("capped" in r.message for r in caplog.records)


# --- idempotency / dismissal safety ------------------------------------------


async def test_rerunning_does_not_duplicate_existing_proposals():
    curation = FakeCuration(clusters=[
        _Cluster("project_x", _Ref("keep", "f"), [_Ref("dup1", "f")], 0.98),
    ])
    engine, driver = _engine(curation=curation)
    first = await engine.propose()
    second = await engine.propose()
    assert first.merges_proposed == 1
    assert second.merges_proposed == 0 and second.already_known == 1


async def test_persist_never_touches_an_existing_proposals_status():
    # The critical anti-nag invariant: a dismissed suggestion must not be resurrected nightly.
    engine, driver = _engine(curation=FakeCuration(clusters=[
        _Cluster("project_x", _Ref("keep", "f"), [_Ref("dup1", "f")], 0.98),
    ]))
    await engine.propose()
    merge_query = next(q[0] for q in driver.queries if "MERGE (p:CurationProposal" in q[0])
    assert "ON CREATE SET" in merge_query
    assert "ON MATCH" not in merge_query, "ON MATCH would overwrite a human's dismissal"


async def test_propose_issues_no_mutating_query_against_knowledge():
    engine, driver = _engine(
        curation=FakeCuration(clusters=[
            _Cluster("project_x", _Ref("keep", "f"), [_Ref("dup1", "f")], 0.98)]),
        cluster_resolver=lambda p: "trading",
    )
    await engine.propose()
    for query, _params in driver.queries:
        if "CurationProposal" in query:
            continue  # writing proposals is the point; they are not knowledge
        lowered = query.lower()
        assert " set " not in lowered and "delete" not in lowered and "merge " not in lowered


# --- promotion target scope --------------------------------------------------


def test_single_cluster_promotes_to_that_cluster():
    engine, _ = _engine(cluster_resolver=lambda pid: "trading")
    assert engine._target_scope(["acme-api", "acme-sim"]) == "cluster_trading"


def test_multiple_clusters_promote_to_global():
    mapping = {"acme-api": "trading", "acme-flow": "infra"}
    engine, _ = _engine(cluster_resolver=mapping.get)
    assert engine._target_scope(["acme-api", "acme-flow"]) == "global"


def test_mixed_clustered_and_unclustered_promotes_to_global():
    mapping = {"acme-api": "trading", "loner": None}
    engine, _ = _engine(cluster_resolver=mapping.get)
    assert engine._target_scope(["acme-api", "loner"]) == "global"


def test_single_project_is_not_promotable():
    engine, _ = _engine(cluster_resolver=lambda pid: "trading")
    assert engine._target_scope(["acme-api"]) is None


def test_no_resolver_falls_back_to_global():
    engine, _ = _engine()
    assert engine._target_scope(["a", "b"]) == "global"


def test_a_broken_resolver_does_not_abort_the_scan():
    def boom(pid):
        raise RuntimeError("registry unreadable")

    engine, _ = _engine(cluster_resolver=boom)
    assert engine._target_scope(["a", "b"]) == "global"


async def test_promotion_proposal_targets_the_cluster_and_names_the_projects():
    driver = FakeDriver(promotion_rows=[
        {"name": "TimescaleDB", "label": "Tool",
         "scopes": ["project_acme-api", "project_acme-sim", "project_acme-data"]},
    ])
    engine, driver = _engine(driver=driver, cluster_resolver=lambda pid: "trading")
    run = await engine.propose()
    assert run.promotions_proposed == 1
    call = next(q[1] for q in driver.queries
                if "MERGE (p:CurationProposal" in q[0] and q[1]["kind"] == "promote")
    assert call["key"] == "promote:TimescaleDB:cluster_trading"
    assert call["props"]["target_scope"] == "cluster_trading"
    assert call["props"]["knowledge_type"] == "tool"
    assert "acme-api" in call["rationale"] and "acme-sim" in call["rationale"]


async def test_promotion_query_excludes_archived_and_invalidated_entities():
    driver = FakeDriver(promotion_rows=[])
    engine, driver = _engine(driver=driver)
    await engine.propose()
    q = next(query for query, _ in driver.queries if "WITH n.name AS name" in query)
    assert "n.invalid_at IS NULL" in q
    assert "coalesce(n.archived, false) = false" in q


# --- review inbox ------------------------------------------------------------


async def test_dismiss_marks_the_proposal_and_reports_it_will_not_return():
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "merge:a:b", "kind": "merge", "status": "open", "rationale": "r",
        "created_at": NOW.isoformat(),
    }})
    engine, driver = _engine(driver=driver)
    out = await engine.dismiss("p1")
    assert out.ok and ("p1", "dismissed") in driver.status_writes
    assert "not be proposed again" in out.detail


async def test_dismiss_unknown_proposal_is_a_clean_failure():
    engine, _ = _engine()
    out = await engine.dismiss("nope")
    assert not out.ok and "not found" in out.detail


async def test_list_proposals_filters_by_status():
    driver = FakeDriver(proposals={
        "p1": {"uuid": "p1", "key": "k1", "kind": "merge", "status": "open",
               "rationale": "", "created_at": NOW.isoformat()},
        "p2": {"uuid": "p2", "key": "k2", "kind": "merge", "status": "dismissed",
               "rationale": "", "created_at": NOW.isoformat()},
    })
    engine, _ = _engine(driver=driver)
    assert [p.uuid for p in await engine.list_proposals(status="open")] == ["p1"]


# --- applying ----------------------------------------------------------------


async def test_applying_a_merge_delegates_to_the_verified_curation_path():
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "merge:keep:dup", "kind": "merge", "status": "open",
        "rationale": "", "created_at": NOW.isoformat(),
        "canonical_uuid": "keep", "duplicate_uuid": "dup",
    }})
    curation = FakeCuration()
    engine, driver = _engine(driver=driver, curation=curation)
    out = await engine.apply("p1")
    assert out.ok
    assert curation.merged == [("keep", "dup")], "must reuse merge_duplicate (backup + verify)"
    assert ("p1", "applied") in driver.status_writes
    assert "backup" in out.detail


async def test_a_failed_merge_leaves_the_proposal_open():
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "merge:keep:dup", "kind": "merge", "status": "open",
        "rationale": "", "created_at": NOW.isoformat(),
        "canonical_uuid": "keep", "duplicate_uuid": "dup",
    }})
    engine, driver = _engine(driver=driver, curation=FakeCuration(merge_ok=False))
    out = await engine.apply("p1")
    assert not out.ok
    assert not driver.status_writes, "a failed apply must not mark the proposal applied"


async def test_applying_a_promotion_without_a_statement_refuses_and_asks():
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "promote:TimescaleDB:cluster_trading", "kind": "promote",
        "status": "open", "rationale": "", "created_at": NOW.isoformat(),
        "name": "TimescaleDB", "target_scope": "cluster_trading",
    }})
    engine, driver = _engine(driver=driver)
    out = await engine.apply("p1")
    assert not out.ok and out.needs_statement
    assert "synthesis, not relocation" in out.detail
    assert not driver.status_writes


async def test_applying_a_promotion_writes_through_the_protected_write_path():
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "promote:TimescaleDB:cluster_trading", "kind": "promote",
        "status": "open", "rationale": "", "created_at": NOW.isoformat(),
        "name": "TimescaleDB", "target_scope": "cluster_trading",
    }})
    seen = {}

    async def remember(content, **kwargs):
        seen["content"] = content
        seen["kwargs"] = kwargs
        return type("W", (), {"scope": "cluster_trading", "outcome": "stored"})()

    engine, driver = _engine(driver=driver, remember=remember)
    out = await engine.apply("p1", statement="TimescaleDB is the standard time-series store.")
    assert out.ok
    assert seen["kwargs"]["cluster"] == "trading"
    assert seen["kwargs"]["source"] == "consolidation"
    assert ("p1", "applied") in driver.status_writes


async def test_promotion_to_global_passes_no_cluster():
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "promote:X:global", "kind": "promote", "status": "open",
        "rationale": "", "created_at": NOW.isoformat(), "name": "X", "target_scope": "global",
    }})
    seen = {}

    async def remember(content, **kwargs):
        seen.update(kwargs)
        return type("W", (), {"scope": "global", "outcome": "stored"})()

    engine, _ = _engine(driver=driver, remember=remember)
    out = await engine.apply("p1", statement="A universal rule.")
    assert out.ok and "cluster" not in seen


async def test_an_already_applied_proposal_cannot_be_applied_twice():
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "merge:a:b", "kind": "merge", "status": "applied",
        "rationale": "", "created_at": NOW.isoformat(),
        "canonical_uuid": "a", "duplicate_uuid": "b",
    }})
    curation = FakeCuration()
    engine, _ = _engine(driver=driver, curation=curation)
    out = await engine.apply("p1")
    assert not out.ok and "already applied" in out.detail
    assert curation.merged == []


async def test_kind_and_status_round_trip_as_enums():
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "merge:a:b", "kind": "merge", "status": "open",
        "rationale": "r", "created_at": NOW.isoformat(),
    }})
    engine, _ = _engine(driver=driver)
    proposals = await engine.list_proposals(status="open")
    assert proposals[0].kind is ProposalKind.MERGE
    assert proposals[0].status is ProposalStatus.OPEN
    assert proposals[0].created_at == NOW


# --- merge safety gate (R8) ---------------------------------------------------
# Every pair below was proposed for merging by the FIRST live run at >=0.97 cosine and is NOT a
# duplicate. Applying them would have destroyed distinct knowledge. Locked in permanently.


@pytest.mark.parametrize("a,b,why", [
    (
        "vectorbt compiles hot paths through llvmlite 0.47.0 as part of the numba stack",
        "vectorbt compiles hot paths through numba 0.65.1 as part of the numba stack",
        "different library and version",
    ),
    (
        "The daily 9:00 beat run had ITB as one of 9 pinned sector ETFs saved in Redis",
        "The daily 9:00 beat run had IBB as one of 9 pinned sector ETFs saved in Redis",
        "different ETF ticker",
    ),
    (
        "Soniox is confirmed for Hebrew real-time STT at $0.12/hr and is the only vendor",
        "Soniox is the only vendor offering native Hebrew-English code-switching",
        "price vs capability",
    ),
    (
        "The similarity floor is 0.72 for recall",
        "The similarity floor is 0.90 for curation",
        "different threshold values",
    ),
    (
        "Acme-API runs on port 8080",
        "Acme-API runs on port 9090",
        "different port",
    ),
])
def test_value_differences_are_never_proposed_for_merge(a, b, why):
    from synapse.core.consolidation_engine import carries_distinct_values

    assert carries_distinct_values(a, b), f"would destroy knowledge: {why}"


@pytest.mark.parametrize("a,b", [
    (
        "BigDecimal is the required type for money in Java",
        "BigDecimal is required as the type for money in Java",
    ),
    (
        "The write pipeline deduplicates before storing",
        "Before storing, the write pipeline deduplicates",
    ),
])
def test_pure_rephrasings_are_still_proposed(a, b):
    from synapse.core.consolidation_engine import carries_distinct_values

    assert not carries_distinct_values(a, b), "a genuine restatement must still be mergeable"


def test_discriminating_tokens_shows_the_reviewer_what_differs():
    from synapse.core.consolidation_engine import discriminating_tokens

    diff = discriminating_tokens("uses llvmlite 0.47.0 here", "uses numba 0.65.1 here")
    assert set(diff) == {"llvmlite", "0.47.0", "numba", "0.65.1"}
    assert "uses" not in diff and "here" not in diff


async def test_unsafe_merges_are_withheld_counted_and_logged(caplog):
    curation = FakeCuration(clusters=[
        _Cluster("project_x",
                 _Ref("keep", "The daily beat had ITB as one of 9 pinned sector ETFs"),
                 [_Ref("dup", "The daily beat had IBB as one of 9 pinned sector ETFs")],
                 0.9834),
    ])
    engine, driver = _engine(curation=curation)
    with caplog.at_level("INFO"):
        run = await engine.propose()
    assert run.merges_proposed == 0
    assert run.unsafe_merges_withheld == 1
    assert any("withholding merge" in r.message for r in caplog.records), "must not be silent"
    assert not [q for q in driver.queries
                if "MERGE (p:CurationProposal" in q[0] and q[1]["kind"] == "merge"]


async def test_a_safe_merge_records_what_differs_for_the_reviewer():
    curation = FakeCuration(clusters=[
        _Cluster("project_x",
                 _Ref("keep", "BigDecimal is the required type for money in Java"),
                 [_Ref("dup", "BigDecimal is required as the type for money in Java")],
                 0.98),
    ])
    engine, driver = _engine(curation=curation)
    run = await engine.propose()
    assert run.merges_proposed == 1 and run.unsafe_merges_withheld == 0
    call = next(q[1] for q in driver.queries if "MERGE (p:CurationProposal" in q[0])
    assert call["props"]["differs_by"], "reviewer must see the wording delta"
    assert "differs by" in call["rationale"].lower()


# The SECOND live run showed the value-only gate was insufficient: these pairs share an identical
# sentence frame but name different things, and all four were proposed for merging at ~0.97.


@pytest.mark.parametrize("a,b,why", [
    (
        "The LangGraph DAG contains a technical analysis node as one of five parallel nodes",
        "The LangGraph DAG contains a fundamental analysis node as one of five parallel nodes",
        "sibling categories: technical vs fundamental",
    ),
    (
        "The Unmanaged-position guard's soft debounce is based on ticks.",
        "The Unmanaged-position guard's hard debounce is based on ticks.",
        "antonyms: soft vs hard",
    ),
    (
        "'flyway_schema_history_watchtower' is an example of the per-service Flyway table",
        "'flyway_schema_history_gateway' is an example of the per-service Flyway table",
        "distinct identifiers",
    ),
    (
        "Sonnet implemented the unified React Flow canvas in Acme-Flow Sprint 1.",
        "Sonnet implemented the unified React Flow canvas and backend reliability tasks in Sprint 1.",
        "superset: the second says strictly more",
    ),
])
def test_identical_sentence_frames_naming_different_things_are_withheld(a, b, why):
    from synapse.core.consolidation_engine import states_different_things

    assert states_different_things(a, b), f"would destroy knowledge — {why}"


@pytest.mark.parametrize("a,b", [
    (
        "BigDecimal is the required type for money in Java",
        "BigDecimal is required as the type for money in Java",
    ),
    (
        "The write pipeline deduplicates before storing",
        "Before storing, the write pipeline deduplicates",
    ),
    (
        "Feature flags must be flipped only after a backtest",
        "A feature flag must only be flipped after backtests",
    ),
])
def test_pure_rephrasings_and_inflections_still_merge(a, b):
    from synapse.core.consolidation_engine import states_different_things

    assert not states_different_things(a, b), "a genuine restatement must stay mergeable"


def test_content_difference_cancels_inflections_and_ignores_function_words():
    from synapse.core.consolidation_engine import content_difference

    only_a, only_b = content_difference(
        "The flags are affecting the run", "A flag affects runs",
    )
    assert only_a == [] and only_b == []


def test_content_difference_reports_the_distinguishing_words():
    from synapse.core.consolidation_engine import content_difference

    only_a, only_b = content_difference(
        "the soft debounce is based on ticks", "the hard debounce is based on ticks",
    )
    assert only_a == ["soft"] and only_b == ["hard"]


# --- semantic adjudication gate ------------------------------------------------
# The lexical gate cannot see role swaps: "A using X (alongside Y)" vs "A using Y (alongside X)"
# has an identical content-word multiset yet states something different. A semantic judge settles
# those, and its absence or failure must never produce an unsafe proposal.


class _Verdict:
    def __init__(self, relation):
        self.relation = relation


class FakeAdjudicator:
    def __init__(self, relation="duplicate", explode=False):
        self.relation = relation
        self.explode = explode
        self.calls: list[tuple[str, str]] = []

    async def adjudicate(self, new_content, existing_fact):
        self.calls.append((new_content, existing_fact))
        if self.explode:
            raise RuntimeError("judge unavailable")
        return _Verdict(self.relation)


def _rephrase_cluster():
    return [_Cluster(
        "project_x", _Ref("keep", "the pipeline deduplicates before storing"),
        [_Ref("dup", "before storing, the pipeline deduplicates")], 0.99,
    )]


async def test_adjudicator_confirming_duplicate_lets_the_merge_through():
    judge = FakeAdjudicator(relation="duplicate")
    engine, _ = _engine(curation=FakeCuration(clusters=_rephrase_cluster()))
    engine._adjudicator = judge
    run = await engine.propose()
    assert run.merges_proposed == 1 and run.adjudicator_rejected == 0
    assert judge.calls, "the judge must actually be consulted"


@pytest.mark.parametrize("relation", ["distinct", "contradiction"])
async def test_adjudicator_rejecting_the_pair_withholds_the_merge(relation):
    engine, driver = _engine(curation=FakeCuration(clusters=_rephrase_cluster()))
    engine._adjudicator = FakeAdjudicator(relation=relation)
    run = await engine.propose()
    assert run.merges_proposed == 0 and run.adjudicator_rejected == 1
    assert not [q for q in driver.queries
                if "MERGE (p:CurationProposal" in q[0] and q[1]["kind"] == "merge"]


async def test_an_adjudicator_outage_fails_CLOSED():
    # Fail-open here would silently drop back to lexical-only and start proposing role swaps.
    engine, _ = _engine(curation=FakeCuration(clusters=_rephrase_cluster()))
    engine._adjudicator = FakeAdjudicator(explode=True)
    run = await engine.propose()
    assert run.merges_proposed == 0 and run.adjudicator_rejected == 1


async def test_without_an_adjudicator_the_lexical_gate_alone_applies():
    engine, _ = _engine(curation=FakeCuration(clusters=_rephrase_cluster()))
    run = await engine.propose()
    assert run.merges_proposed == 1 and run.adjudicator_rejected == 0


async def test_the_adjudicator_is_not_consulted_for_lexically_unsafe_pairs():
    # No point spending a judge call on a pair already known to differ by a value.
    judge = FakeAdjudicator()
    engine, _ = _engine(curation=FakeCuration(clusters=[
        _Cluster("project_x", _Ref("keep", "Acme-API runs on port 8080"),
                 [_Ref("dup", "Acme-API runs on port 9090")], 0.99),
    ]))
    engine._adjudicator = judge
    run = await engine.propose()
    assert run.unsafe_merges_withheld == 1 and judge.calls == []


def test_counting_catches_same_word_set_different_multiplicity():
    from synapse.core.consolidation_engine import states_different_things

    assert states_different_things(
        "The LightGBM + XGBoost ensemble includes XGBoost as a component.",
        "The LightGBM + XGBoost ensemble includes LightGBM as a component.",
    ), "identical word SETS but different counts => different things"


def test_symmetric_relation_permutations_are_lexically_mergeable():
    # Verified against the live graph: "alongside" and "simultaneously with" are symmetric, so
    # which term is the grammatical subject carries no information — these ARE duplicates, and
    # the semantic judge confirmed them as such. The lexical gate must not pre-emptively veto.
    from synapse.core.consolidation_engine import states_different_things

    assert not states_different_things(
        "Walk-forward validation is performed using backtrader (alongside vectorbt)",
        "Walk-forward validation is performed using vectorbt (alongside backtrader)",
    )
    assert not states_different_things(
        "MES feed went silent simultaneously with NQ, ES, and MNQ at 13:00 ET",
        "NQ feed went silent simultaneously with ES, MNQ, and MES at 13:00 ET",
    )


# --- promotion statement quality ----------------------------------------------
# Learned from the first real promotion: a statement that enumerates the projects gets decomposed
# by the extractor into one near-identical fact per project, which then outranks the substantive
# knowledge. It cost 12% MRR on the eval set before being caught.


def _promote_proposal_driver():
    return FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "promote:TimescaleDB:cluster_trading", "kind": "promote",
        "status": "open", "rationale": "", "created_at": NOW.isoformat(),
        "name": "TimescaleDB", "target_scope": "cluster_trading",
        "from_scopes": ["project_acme-api", "project_acme-sim", "project_acme-data",
                        "project_acme-etl"],
    }})


async def test_a_statement_enumerating_the_projects_is_rejected():
    calls = []

    async def remember(content, **kwargs):
        calls.append(content)
        return type("W", (), {"scope": "cluster_trading", "outcome": "stored"})()

    engine, driver = _engine(driver=_promote_proposal_driver(), remember=remember)
    out = await engine.apply("p1", statement=(
        "TimescaleDB is the standard time-series store across the trading projects "
        "(acme-data, acme-api, acme-etl, acme-sim), used for OHLCV bars."
    ))
    assert not out.ok and out.needs_statement
    assert "enumerates" in out.detail
    assert calls == [], "nothing may be written when the statement is rejected"
    assert not driver.status_writes, "the proposal must stay open"


async def test_a_focused_statement_is_accepted():
    calls = []

    async def remember(content, **kwargs):
        calls.append(content)
        return type("W", (), {"scope": "cluster_trading", "outcome": "stored"})()

    engine, _ = _engine(driver=_promote_proposal_driver(), remember=remember)
    out = await engine.apply("p1", statement=(
        "TimescaleDB is the standard time-series store for the trading domain: OHLCV bars and "
        "tick history live in hypertables."
    ))
    assert out.ok and len(calls) == 1


async def test_mentioning_one_or_two_projects_is_still_allowed():
    # Naming a project for context is fine; enumerating the whole set is what causes fan-out.
    async def remember(content, **kwargs):
        return type("W", (), {"scope": "cluster_trading", "outcome": "stored"})()

    engine, _ = _engine(driver=_promote_proposal_driver(), remember=remember)
    out = await engine.apply("p1", statement=(
        "TimescaleDB hypertables need the partition column in the primary key, as acme-api found."
    ))
    assert out.ok


async def test_the_empty_statement_message_warns_against_enumeration():
    engine, _ = _engine(driver=_promote_proposal_driver())
    out = await engine.apply("p1")
    assert out.needs_statement
    assert "do not list the projects" in out.detail


# --- promotion dedup against the WIDER scopes (roadmap item 23) ----------------
# A promotion's target scope is typically EMPTY — that is the point of promoting — so scope-only
# dedup can never notice that `global` already holds the same knowledge. The first real promotion
# stored a duplicate of an existing global fact for exactly this reason.


def test_promotion_dedup_scopes_include_the_tiers_above():
    assert ConsolidationEngine._promotion_dedup_scopes("cluster_trading") == [
        "cluster_trading", "global",
    ]


def test_promotion_to_global_compares_against_global_only():
    assert ConsolidationEngine._promotion_dedup_scopes("global") == ["global"]


def test_promotion_dedup_never_includes_tiers_below():
    # A project-contextualized restatement is legitimately distinct knowledge.
    scopes = ConsolidationEngine._promotion_dedup_scopes("cluster_trading")
    assert not any(s.startswith("project_") for s in scopes)


class _Write:
    def __init__(self, outcome, scope="cluster_trading", reason="", duplicate_of=None):
        self.outcome = type("O", (), {"value": outcome})()
        self.scope = scope
        self.reason = reason
        self.duplicate_of = duplicate_of


def _promo_driver():
    return FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "promote:X:cluster_trading", "kind": "promote", "status": "open",
        "rationale": "", "created_at": NOW.isoformat(), "name": "X",
        "target_scope": "cluster_trading", "from_scopes": ["project_a", "project_b"],
    }})


async def test_apply_promote_passes_the_widened_dedup_scopes():
    seen = {}

    async def remember(content, **kwargs):
        seen.update(kwargs)
        return _Write("stored")

    engine, _ = _engine(driver=_promo_driver(), remember=remember)
    out = await engine.apply("p1", statement="A shared truth about the domain.")
    assert out.ok
    assert seen["dedup_scopes"] == ["cluster_trading", "global"]
    assert "force" not in seen, "promotion must not bypass the quality gates"


async def test_a_redundant_promotion_reports_duplicate_and_stops_being_proposed():
    async def remember(content, **kwargs):
        return _Write("duplicate", duplicate_of="edge-9")

    engine, driver = _engine(driver=_promo_driver(), remember=remember)
    out = await engine.apply("p1", statement="Something global already says.")
    assert out.ok, "a no-op promotion is not a failure"
    assert "already covered" in out.detail and "edge-9" in out.detail
    assert ("p1", "applied") in driver.status_writes, "must not be re-proposed nightly"


async def test_a_rejected_statement_leaves_the_proposal_open_for_a_rewrite():
    async def remember(content, **kwargs):
        return _Write("rejected", reason="did not pass the write-trigger filter")

    engine, driver = _engine(driver=_promo_driver(), remember=remember)
    out = await engine.apply("p1", statement="lol idk something")
    assert not out.ok and out.needs_statement
    assert "rejected this statement" in out.detail
    assert not driver.status_writes, "a rejected promotion must stay open"


# --- contradiction sweep (roadmap item 16, second half) ------------------------
# The write path only judges a NEW write against its neighbours, so knowledge already sitting side
# by side was never re-examined. The corpus shows the cost: 7 Contradicts edges across 3,039 facts.


def _review_pair(a_uuid, a_fact, b_uuid, b_fact, sim=0.93):
    return _Pair("project_x", _Ref(a_uuid, a_fact), _Ref(b_uuid, b_fact), sim)


async def test_the_sweep_proposes_only_pairs_the_judge_calls_contradictory():
    curation = FakeCuration(review_pairs=[
        _review_pair("a1", "Acme-API runs on port 8080", "b1", "Acme-API runs on port 9090"),
        _review_pair("a2", "the retry budget is 3 attempts", "b2", "the retry budget is 5 attempts"),
    ])
    engine, driver = _engine(curation=curation)
    engine._adjudicator = FakeAdjudicator(relation="contradiction")
    run = await engine.propose(max_merges=0, max_promotions=0)
    assert run.contradictions_proposed == 2

    engine2, _ = _engine(curation=curation)
    engine2._adjudicator = FakeAdjudicator(relation="distinct")
    run2 = await engine2.propose(max_merges=0, max_promotions=0)
    assert run2.contradictions_proposed == 0


async def test_the_lexical_merge_gate_does_not_apply_to_contradictions():
    # A value difference ("port 8080" vs "port 9090") is exactly what a contradiction LOOKS like.
    # The gate exists to stop destructive merges; a contradiction proposal destroys nothing.
    curation = FakeCuration(review_pairs=[
        _review_pair("a1", "Acme-API runs on port 8080", "b1", "Acme-API runs on port 9090"),
    ])
    engine, _ = _engine(curation=curation)
    engine._adjudicator = FakeAdjudicator(relation="contradiction")
    run = await engine.propose(max_merges=0, max_promotions=0)
    assert run.contradictions_proposed == 1
    assert run.unsafe_merges_withheld == 0


async def test_without_an_adjudicator_the_sweep_is_a_no_op_not_a_guess():
    # Flagging random similar pairs as contradictions would poison the review queue.
    curation = FakeCuration(review_pairs=[
        _review_pair("a1", "one claim", "b1", "another claim"),
    ])
    engine, _ = _engine(curation=curation)
    run = await engine.propose(max_merges=0, max_promotions=0)
    assert run.contradictions_proposed == 0


async def test_a_judge_failure_skips_the_pair_and_continues_the_sweep():
    calls = {"n": 0}

    class Flaky(FakeAdjudicator):
        async def adjudicate(self, new_content, existing_fact):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("judge blipped")
            return _Verdict("contradiction")

    curation = FakeCuration(review_pairs=[
        _review_pair("a1", "the floor is 0.72", "b1", "the floor is 0.80"),
        _review_pair("a2", "the cap is 40 pairs", "b2", "the cap is 60 pairs"),
    ])
    engine, _ = _engine(curation=curation)
    engine._adjudicator = Flaky()
    run = await engine.propose(max_merges=0, max_promotions=0)
    assert run.contradictions_proposed == 1, "the second pair must still be judged"


async def test_the_sweep_is_capped():
    curation = FakeCuration(review_pairs=[
        _review_pair(f"a{i}", f"the timeout is {i} seconds", f"b{i}", f"the timeout is {i + 90} seconds")
        for i in range(10)
    ])
    engine, _ = _engine(curation=curation)
    engine._adjudicator = FakeAdjudicator(relation="contradiction")
    run = await engine.propose(max_merges=0, max_promotions=0, max_contradictions=3)
    assert run.contradictions_proposed == 3


async def test_a_contradiction_proposal_cannot_be_applied_automatically():
    # Resolving one means deciding which fact is true — a judgement about the world, not the graph.
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "contradiction:a1:b1", "kind": "contradiction", "status": "open",
        "rationale": "", "created_at": NOW.isoformat(),
        "canonical_uuid": "a1", "duplicate_uuid": "b1",
    }})
    curation = FakeCuration()
    engine, driver = _engine(driver=driver, curation=curation)
    out = await engine.apply("p1")
    assert not out.ok
    assert "cannot be applied automatically" in out.detail
    assert "update" in out.detail
    assert curation.merged == [], "it must never fall through to a merge"
    assert not driver.status_writes, "the proposal stays open until a human resolves it"


async def test_a_contradiction_proposal_can_still_be_dismissed():
    driver = FakeDriver(proposals={"p1": {
        "uuid": "p1", "key": "contradiction:a1:b1", "kind": "contradiction", "status": "open",
        "rationale": "", "created_at": NOW.isoformat(),
    }})
    engine, driver = _engine(driver=driver)
    out = await engine.dismiss("p1")
    assert out.ok and ("p1", "dismissed") in driver.status_writes


async def test_the_sweep_requires_a_value_difference_the_mirror_of_the_merge_gate():
    # A contradiction is the same single-valued thing given two different values, so a value
    # difference is a PREREQUISITE here — the very signal that DISQUALIFIES a merge. Verified on
    # real data: the local judge called "handles the sentiment node" vs "handles the technical node"
    # contradictory, when both are simply true. Set members cannot conflict.
    curation = FakeCuration(review_pairs=[
        _review_pair("a1", "Claude Haiku handles the sentiment analysis node",
                     "b1", "Claude Haiku handles the technical analysis node"),
        _review_pair("a2", "The 567K-trade exit study informed the ratchet design",
                     "b2", "The 11K-trade exit study informed the ratchet design"),
    ])
    engine, _ = _engine(curation=curation)
    judge = FakeAdjudicator(relation="contradiction")
    engine._adjudicator = judge
    run = await engine.propose(max_merges=0, max_promotions=0)
    assert run.contradictions_proposed == 1, "only the numeric conflict is a real candidate"
    assert len(judge.calls) == 1, "the set-member pair must not even cost a judge call"


def test_the_merge_gate_and_the_contradiction_filter_are_exact_opposites():
    # Same predicate, opposite polarity — worth asserting so a future edit cannot align them.
    from synapse.core.consolidation_engine import carries_distinct_values

    value_diff = ("Acme-API runs on port 8080", "Acme-API runs on port 9090")
    set_member = ("Haiku handles the sentiment node", "Haiku handles the technical node")
    assert carries_distinct_values(*value_diff)        # contradiction candidate, merge-disqualified
    assert not carries_distinct_values(*set_member)    # mergeable shape, cannot contradict
