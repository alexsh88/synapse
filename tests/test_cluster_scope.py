"""Cluster scope tier — the domain layer between project and global (research §0).

Measuring the live graph showed the scope hierarchy was degenerate: 95.5% of nodes in
`project_*`, 4.5% in `global`, `agent_*` empty, and only 30 of 3,011 edges (1.0%) crossing
projects. Domain-general-but-not-universal knowledge had nowhere to live, so it stayed siloed.
"""

from __future__ import annotations

import pytest

from synapse.core.registry import _NON_DOMAIN_CLUSTERS, cluster_of
from synapse.core.retrieval_engine import Fact, RetrievalEngine
from synapse.core.schema import Scope


# --- Scope.compose ------------------------------------------------------------


def test_cluster_scope_naming():
    assert Scope.cluster("trading") == "cluster_trading"


def test_compose_orders_widest_to_narrowest():
    assert Scope.compose("acme-api", cluster="trading") == [
        "global", "cluster_trading", "project_acme-api",
    ]


def test_compose_without_cluster_is_unchanged():
    # Backwards compatibility: every existing caller keeps its exact behaviour.
    assert Scope.compose("acme-store") == ["global", "project_acme-store"]
    assert Scope.compose() == ["global"]
    assert Scope.compose("gf", "planner") == ["global", "project_gf", "agent_planner"]


def test_compose_full_lattice():
    assert Scope.compose("acme-api", "reviewer", cluster="trading") == [
        "global", "cluster_trading", "project_acme-api", "agent_reviewer",
    ]


# --- registry resolution ------------------------------------------------------


def test_bookkeeping_clusters_are_not_domains():
    # "added" is add_connected()'s default and "standalone" means no shared domain — neither
    # should create a cluster tier, or unrelated projects would pool knowledge together.
    for junk in ("added", "standalone", "none", ""):
        assert junk in _NON_DOMAIN_CLUSTERS


def test_cluster_of_unknown_project_is_none():
    assert cluster_of("no-such-project-xyz") is None


# --- retrieval composition ----------------------------------------------------


class _Searcher:
    def __init__(self):
        self.last = None

    async def search(self, query, scopes, limit, center_node_uuid):
        self.last = {"scopes": scopes}
        return []


class _Queries:
    async def nodes_by_label(self, labels, scopes, limit):
        return []

    async def degrees(self, uuids):
        return {}

    async def node_confidence(self, uuids):
        return {}


async def test_recall_composes_the_cluster_tier():
    searcher = _Searcher()
    engine = RetrievalEngine(searcher, _Queries(), redis=None,
                             cluster_resolver=lambda pid: "trading")
    await engine.recall("q", project_id="acme-api")
    assert searcher.last["scopes"] == ["global", "cluster_trading", "project_acme-api"]


async def test_recall_omits_the_tier_when_the_project_has_no_domain():
    searcher = _Searcher()
    engine = RetrievalEngine(searcher, _Queries(), redis=None,
                             cluster_resolver=lambda pid: None)
    await engine.recall("q", project_id="loner")
    assert searcher.last["scopes"] == ["global", "project_loner"]


async def test_recall_without_a_resolver_behaves_exactly_as_before():
    searcher = _Searcher()
    engine = RetrievalEngine(searcher, _Queries(), redis=None)
    await engine.recall("q", project_id="acme-api")
    assert searcher.last["scopes"] == ["global", "project_acme-api"]


async def test_a_broken_registry_never_breaks_recall():
    def boom(pid):
        raise RuntimeError("registry unreadable")

    searcher = _Searcher()
    engine = RetrievalEngine(searcher, _Queries(), redis=None, cluster_resolver=boom)
    await engine.recall("q", project_id="acme-api")
    assert searcher.last["scopes"] == ["global", "project_acme-api"]


# --- write-side placement -----------------------------------------------------


def test_write_scope_resolution_precedence():
    from synapse.core.write_pipeline import TriageVerdict, WritePipeline

    pipeline = WritePipeline(graphiti=None, embedder=None, index=None, triage=None)
    v = TriageVerdict(worth_storing=True)
    # cluster beats project (the caller is deliberately widening to the whole domain)...
    assert pipeline._resolve_scope(v, "acme-api", None, "trading") == "cluster_trading"
    # ...but agent_role stays the narrowest scope of all.
    assert pipeline._resolve_scope(v, "acme-api", "planner", "trading") == "agent_planner"
    # and with no cluster, nothing changes.
    assert pipeline._resolve_scope(v, "acme-api", None, None) == "project_acme-api"
    assert pipeline._resolve_scope(v, None, None, None) == "global"


@pytest.mark.parametrize("scope,expected", [
    ("cluster:trading", "trading"),
    ("cluster_trading", "trading"),
    ("CLUSTER:Trading", "trading"),
    ("project:acme-api", None),
    ("global", None),
    (None, None),
])
def test_mcp_scope_arg_parses_cluster(scope, expected):
    from synapse.mcp.tools import _scope_to_cluster

    assert _scope_to_cluster(scope) == expected


async def test_mcp_remember_routes_a_cluster_write_without_a_project_scope():
    from synapse.mcp import tools as t

    class Eng:
        def __init__(self):
            self.kw = None

        async def remember(self, content, *, knowledge_type=None, project_id=None, cluster=None):
            self.kw = {"project_id": project_id, "cluster": cluster}

            class R:
                outcome = type("O", (), {"value": "stored"})()
                knowledge_type = "lesson"
                scope = "cluster_trading"
                episode_uuid = "e1"
                entities: list = []
                facts: list = []
                duplicate_of = contradicts = None
                reason = ""
                degraded = False
                facts_extracted = 0
                redactions: list = []

            return R()

    eng = Eng()
    await t.remember(eng, "acme-api", "broker historical volume is in lots of 100",
                     scope="cluster:trading")
    assert eng.kw == {"project_id": None, "cluster": "trading"}


# --- edge-name canonicalization (research §2.2) -------------------------------
# The live graph reached 535 distinct edge names over 3,018 edges, with 328 names used exactly
# once. Canonicalization must fold true variants and refuse everything else.


@pytest.mark.parametrize("raw,expected", [
    ("APPLIES_TO", "AppliesTo"),
    ("applies_to", "AppliesTo"),
    ("AppliesTo", "AppliesTo"),
    ("APPLIED_TO", "AppliesTo"),
    ("APPLIED_IN", "AppliesTo"),
    ("USED_IN", "UsedIn"),
    ("RELATED_TO", "RelatedTo"),
    ("SUPERSEDES", "Supersedes"),
    ("HAS_GOTCHA", "HasGotcha"),
    ("DERIVED_FROM", "DerivedFrom"),
    ("conflicts_with", "Contradicts"),
    ("CITES", "References"),
])
def test_true_variants_fold_onto_the_schema_type(raw, expected):
    from synapse.core.schema import canonical_edge_name

    assert canonical_edge_name(raw) == expected


def test_uses_is_never_folded_into_usedin_because_direction_is_inverted():
    # UsedIn runs Pattern->Project; USES runs the other way. Folding would invert 134 live edges.
    # Since roadmap item 24 gave the relation its OWN type, it canonicalizes to Uses — the point
    # is that it must never become UsedIn.
    from synapse.core.schema import canonical_edge_name

    assert canonical_edge_name("USES") == "Uses"
    assert canonical_edge_name("USES") != "UsedIn"


@pytest.mark.parametrize("specific", [
    "CAUSED_BY", "FIXES", "ENFORCES", "CONTAINS", "PART_OF", "CALLS",
    "PINNED_SECTOR_ETF", "IS_IMPLEMENTED_SERVICE_OF", "DETECTED_UNTRACKED", "TRIGGERS",
])
def test_distinct_relations_are_never_flattened_into_a_generic_type(specific):
    # Collapsing a specific relation into RelatedTo destroys meaning — the same class of mistake
    # as merging two facts that merely share a sentence frame. Roadmap item 24 gave most of these
    # their OWN specific type; the invariant is that none is ever answered with a generic one.
    from synapse.core.schema import canonical_edge_name

    assert canonical_edge_name(specific) not in ("RelatedTo", "AppliesTo")


def test_canonicalization_is_idempotent_and_safe_on_empty():
    from synapse.core.schema import canonical_edge_name

    once = canonical_edge_name("APPLIED_TO")
    assert canonical_edge_name(once) == once
    assert canonical_edge_name("") == ""
    assert canonical_edge_name(None) is None


def test_every_schema_type_is_its_own_canonical_form():
    from synapse.core.schema import EDGE_TYPES, canonical_edge_name

    for name in EDGE_TYPES:
        assert canonical_edge_name(name) == name


def test_no_synonym_maps_to_a_non_schema_name():
    from synapse.core.schema import EDGE_NAME_SYNONYMS, EDGE_TYPES

    for source, target in EDGE_NAME_SYNONYMS.items():
        assert target in EDGE_TYPES, f"{source} maps to unknown type {target}"


async def test_write_path_renames_extracted_edges_to_canonical_names():
    from synapse.core.write_pipeline import WritePipeline

    class _Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **params):
            self.calls.append((query, params))

            class _R:
                records = []

            return _R()

    class _G:
        def __init__(self, driver):
            self.driver = driver

    class _Edge:
        def __init__(self, uuid, name):
            self.uuid = uuid
            self.name = name

    driver = _Driver()
    pipeline = WritePipeline(graphiti=_G(driver), embedder=None, index=None, triage=None)
    result = type("R", (), {"edges": [
        _Edge("e1", "APPLIED_TO"),     # folds
        _Edge("e2", "PINNED_SECTOR_ETF"),   # kept — project-specific, no schema type
        _Edge("e3", "AppliesTo"),      # already canonical
    ]})()
    await pipeline._canonicalize_edge_names(result)

    rename_calls = [p for q, p in driver.calls if "name_before_canonicalization" in q]
    assert len(rename_calls) == 1
    renames = rename_calls[0]["renames"]
    assert renames == [{"uuid": "e1", "name": "AppliesTo", "was": "APPLIED_TO"}]


async def test_canonicalization_failure_never_breaks_a_write():
    from synapse.core.write_pipeline import WritePipeline

    class _Boom:
        async def execute_query(self, query, **params):
            raise RuntimeError("neo4j down")

    class _G:
        driver = _Boom()

    pipeline = WritePipeline(graphiti=_G(), embedder=None, index=None, triage=None)
    result = type("R", (), {"edges": [type("E", (), {"uuid": "e1", "name": "APPLIED_TO"})()]})()
    await pipeline._canonicalize_edge_names(result)   # must not raise


# --- EDGE_TYPES extension (research §2.2, roadmap item 24) ---------------------
# The recurring residual was real vocabulary the schema did not model. These 12 types are that
# tail; the counts below are the live-graph volumes they consolidate.


@pytest.mark.parametrize("raw,expected", [
    # Uses (163 edges) — dependency direction
    ("USES", "Uses"), ("USES_TOOL", "Uses"), ("BUILT_WITH", "Uses"), ("BUILT_ON", "Uses"),
    # Implements (27)
    ("IMPLEMENTS", "Implements"), ("IMPLEMENTED", "Implements"),
    # DefinedIn (57) — containment of a definition
    ("IMPLEMENTED_IN", "DefinedIn"), ("DEFINED_IN", "DefinedIn"), ("LOCATED_IN", "DefinedIn"),
    # PartOf (35) / Contains (21) — inverses of each other
    ("PART_OF", "PartOf"), ("BELONGS_TO", "PartOf"),
    ("IS_IMPLEMENTED_SERVICE_OF", "PartOf"),
    ("CONTAINS", "Contains"), ("INCLUDES", "Contains"),
    # causal family
    ("CAUSED", "Causes"), ("CAUSED_BY", "CausedBy"), ("FIXES", "Fixes"),
    # Enforces (21)
    ("ENFORCES", "Enforces"), ("GATES", "Enforces"),
    # weak influence
    ("AFFECTED", "Affects"),
    ("CONTRIBUTED_TO", "ContributesTo"), ("CONTRIBUTES_TO", "ContributesTo"),
    # code-level
    ("CALLS", "Calls"),
])
def test_recurring_residual_folds_onto_the_new_types(raw, expected):
    from synapse.core.schema import canonical_edge_name

    assert canonical_edge_name(raw) == expected


def test_causes_and_caused_by_stay_separate_types():
    # They are INVERSES. Folding one into the other would require swapping endpoints, so both
    # exist instead and no edge has to be surgically reversed.
    from synapse.core.schema import EDGE_TYPES, canonical_edge_name

    assert canonical_edge_name("CAUSED") == "Causes"
    assert canonical_edge_name("CAUSED_BY") == "CausedBy"
    assert "Causes" in EDGE_TYPES and "CausedBy" in EDGE_TYPES


def test_uses_and_usedin_remain_distinct_inverses():
    from synapse.core.schema import canonical_edge_name

    assert canonical_edge_name("USES") == "Uses"
    assert canonical_edge_name("USED_IN") == "UsedIn"


def test_partof_and_contains_remain_distinct_inverses():
    from synapse.core.schema import canonical_edge_name

    assert canonical_edge_name("PART_OF") == "PartOf"
    assert canonical_edge_name("CONTAINS") == "Contains"


@pytest.mark.parametrize("still_untyped", ["TRIGGERS", "DETECTED_UNTRACKED", "PINNED_SECTOR_ETF"])
def test_project_specific_and_marginal_relations_are_still_left_alone(still_untyped):
    # Below the bar (8 edges) or project-specific, so not general schema vocabulary. Keeping their
    # own names is the honest outcome — better than a wrong fold.
    from synapse.core.schema import canonical_edge_name

    assert canonical_edge_name(still_untyped) == still_untyped


def test_every_new_type_has_an_extraction_docstring_stating_direction():
    # Docstrings are load-bearing: the extractor reads them to choose a relation.
    from synapse.core.schema import EDGE_TYPES

    new = ["Uses", "Implements", "DefinedIn", "PartOf", "Contains", "Causes", "CausedBy",
           "Fixes", "Enforces", "Affects", "ContributesTo", "Calls"]
    for name in new:
        doc = (EDGE_TYPES[name].__doc__ or "")
        assert doc.strip(), f"{name} has no docstring"
        assert "irection" in doc or "inverse" in doc.lower(), f"{name} does not state direction"


def test_edge_type_map_only_references_real_types():
    from synapse.core.schema import EDGE_TYPES, EDGE_TYPE_MAP

    for pair, types in EDGE_TYPE_MAP.items():
        assert len(pair) == 2
        for t in types:
            assert t in EDGE_TYPES, f"{pair} allows unknown type {t}"


def test_edge_type_map_covers_the_new_types():
    from synapse.core.schema import EDGE_TYPE_MAP

    allowed = {t for types in EDGE_TYPE_MAP.values() for t in types}
    for name in ("Uses", "Implements", "DefinedIn", "PartOf", "Contains", "Causes",
                 "CausedBy", "Fixes", "Enforces", "Affects", "ContributesTo", "Calls"):
        assert name in allowed, f"{name} is defined but no label pair permits it"


def test_entity_entity_catchall_still_allows_relatedto():
    # The generic fallback must survive, or unclassifiable relations have nowhere to go.
    from synapse.core.schema import EDGE_TYPE_MAP

    assert "RelatedTo" in EDGE_TYPE_MAP[("Entity", "Entity")]
