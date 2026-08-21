"""Unit tests for the write pipeline (plan Part 4).

All external services (Claude triage, BGE-M3 embedder, Neo4j vector index, the
Graphiti graph) are faked, so these test the pipeline's *logic* — the write
trigger, scope resolution, dedup threshold, and contradiction flagging — without
touching the network. A live end-to-end run lives in scripts/schema_smoke.py.
"""

from __future__ import annotations

from synapse.core.redaction import redact
from synapse.core.schema import ENTITY_TYPES
from synapse.core.write_pipeline import (
    Adjudication,
    NearestFact,
    Neo4jVectorIndex,
    Outcome,
    Relation,
    TriageVerdict,
    WritePipeline,
    _content_hash,
)

# --- fakes -------------------------------------------------------------------


class FakeEmbedder:
    async def create(self, input_data: str) -> list[float]:
        return [0.1] * 1024


class FakeIndex:
    def __init__(self, nearest: NearestFact | None = None):
        self._nearest = nearest

    async def nearest(self, vec, scopes):
        return self._nearest


class FakeTriage:
    def __init__(self, verdict: TriageVerdict, adjudication: Adjudication | None = None):
        self._verdict = verdict
        self._adj = adjudication or Adjudication(relation=Relation.DISTINCT)
        self.adjudicated = False

    async def classify(self, content, hint_type):
        return self._verdict

    async def adjudicate(self, new_content, existing_fact):
        self.adjudicated = True
        return self._adj


class _Node:
    def __init__(self, name):
        self.name = name


class _Edge:
    def __init__(self, name, fact):
        self.name = name
        self.fact = fact


class _Episode:
    uuid = "ep-1"


class _AddResult:
    episode = _Episode()
    nodes = [_Node("Acme-Store"), _Node("SQLite")]
    edges = [_Edge("AppliesTo", "Acme-Store uses SQLite")]


class FakeGraphiti:
    def __init__(self):
        self.calls: list[dict] = []

    async def add_episode(self, **kwargs):
        self.calls.append(kwargs)
        return _AddResult()


def make_pipeline(*, verdict, nearest=None, adjudication=None):
    graphiti = FakeGraphiti()
    triage = FakeTriage(verdict, adjudication)
    pipeline = WritePipeline(
        graphiti=graphiti,
        embedder=FakeEmbedder(),
        index=FakeIndex(nearest),
        triage=triage,
        dedup_threshold=0.9,
        relate_floor=0.75,
    )
    return pipeline, graphiti, triage


STORABLE = TriageVerdict(worth_storing=True, knowledge_type="decision", is_global=False, confidence=0.8)


# --- tests -------------------------------------------------------------------


async def test_new_knowledge_stored():
    pipeline, graphiti, _ = make_pipeline(verdict=STORABLE, nearest=None)
    result = await pipeline.remember("We chose SQLite for Acme-Store.", project_id="acme-store")

    assert result.outcome is Outcome.STORED
    assert result.scope == "project_acme-store"
    assert result.knowledge_type == "decision"
    assert result.confidence == 0.8
    assert len(graphiti.calls) == 1  # actually stored


async def test_entities_extracted_and_schema_forwarded():
    pipeline, graphiti, _ = make_pipeline(verdict=STORABLE, nearest=None)
    result = await pipeline.remember("We chose SQLite for Acme-Store.", project_id="acme-store")

    # The pipeline returns Graphiti's extracted entities/facts...
    assert result.entities == ["Acme-Store", "SQLite"]
    assert result.facts == ["Acme-Store uses SQLite"]
    assert result.episode_uuid == "ep-1"
    # ...and forwards the Synapse schema so extraction is typed.
    assert graphiti.calls[0]["entity_types"] is ENTITY_TYPES
    assert graphiti.calls[0]["group_id"] == "project_acme-store"


async def test_duplicate_caught_by_threshold():
    near = NearestFact(uuid="edge-9", fact="Acme-Store uses SQLite", score=0.95)
    pipeline, graphiti, triage = make_pipeline(verdict=STORABLE, nearest=near)
    result = await pipeline.remember("Acme-Store's backend is SQLite.", project_id="acme-store")

    assert result.outcome is Outcome.DUPLICATE
    assert result.duplicate_of == "edge-9"
    assert len(graphiti.calls) == 0  # not re-stored
    assert triage.adjudicated is False  # >=0.9 short-circuits before adjudication


async def test_duplicate_caught_by_adjudication():
    # In the gray zone (>= floor, < dedup) Haiku adjudicates duplicate.
    near = NearestFact(uuid="edge-7", fact="Acme-Store uses SQLite", score=0.80)
    pipeline, graphiti, triage = make_pipeline(
        verdict=STORABLE, nearest=near, adjudication=Adjudication(relation=Relation.DUPLICATE)
    )
    result = await pipeline.remember("The Acme-Store DB is SQLite.", project_id="acme-store")

    assert result.outcome is Outcome.DUPLICATE
    assert result.duplicate_of == "edge-7"
    assert triage.adjudicated is True
    assert len(graphiti.calls) == 0


async def test_contradiction_flagged_and_stored():
    near = NearestFact(uuid="edge-3", fact="Acme-Store uses SQLite", score=0.82)
    pipeline, graphiti, triage = make_pipeline(
        verdict=STORABLE, nearest=near, adjudication=Adjudication(relation=Relation.CONTRADICTION)
    )
    result = await pipeline.remember("Acme-Store switched to PostgreSQL.", project_id="acme-store")

    assert result.outcome is Outcome.CONTRADICTION
    assert result.contradicts == "edge-3"
    assert len(graphiti.calls) == 1  # new truth is still stored (temporal supersede)


async def test_distinct_in_gray_zone_stores_normally():
    near = NearestFact(uuid="edge-5", fact="Acme-Store uses FastAPI", score=0.78)
    pipeline, graphiti, triage = make_pipeline(
        verdict=STORABLE, nearest=near, adjudication=Adjudication(relation=Relation.DISTINCT)
    )
    result = await pipeline.remember("Acme-Store monetizes with fail offers.", project_id="acme-store")

    assert result.outcome is Outcome.STORED
    assert result.contradicts is None
    assert triage.adjudicated is True
    assert len(graphiti.calls) == 1


async def test_noise_rejected_by_write_trigger():
    noise = TriageVerdict(worth_storing=False, knowledge_type="entity", reason="raw chatter")
    pipeline, graphiti, _ = make_pipeline(verdict=noise, nearest=None)
    result = await pipeline.remember("ok thanks, let me check that real quick", project_id="acme-store")

    assert result.outcome is Outcome.REJECTED
    assert len(graphiti.calls) == 0  # never reaches the store


async def test_force_bypasses_write_trigger():
    noise = TriageVerdict(worth_storing=False, knowledge_type="lesson")
    pipeline, graphiti, _ = make_pipeline(verdict=noise, nearest=None)
    result = await pipeline.remember("store this anyway", project_id="acme-store", force=True)

    assert result.outcome is Outcome.STORED
    assert len(graphiti.calls) == 1


async def test_explicit_project_wins_over_is_global():
    # An explicit project_id ALWAYS wins, even if triage guesses is_global (R5; the
    # docstring's "explicit project wins"). Promotion to global is a curation action,
    # not an unreliable per-write LLM guess — this prevents cross-project leakage.
    glob = TriageVerdict(worth_storing=True, knowledge_type="pattern", is_global=True)
    pipeline, _, _ = make_pipeline(verdict=glob, nearest=None)
    result = await pipeline.remember("A claudeModel mutation bug in regenWithModel.", project_id="acme-bot")
    assert result.scope == "project_acme-bot"


async def test_scope_global_only_when_no_project():
    # is_global no longer overrides scope; with no project context, knowledge is global.
    glob = TriageVerdict(worth_storing=True, knowledge_type="pattern", is_global=True)
    pipeline, _, _ = make_pipeline(verdict=glob, nearest=None)
    result = await pipeline.remember("Always back up keystores in three places.")
    assert result.scope == "global"


async def test_scope_defaults_to_project():
    pipeline, _, _ = make_pipeline(verdict=STORABLE, nearest=None)
    result = await pipeline.remember("Acme-Store ships Android only.", project_id="acme-store")
    assert result.scope == "project_acme-store"


async def test_scope_agent_role():
    pipeline, _, _ = make_pipeline(verdict=STORABLE, nearest=None)
    result = await pipeline.remember("Planner prefers risk-first sequencing.", agent_role="planner")
    assert result.scope == "agent_planner"


# --- WP-B item 1: triage fails closed --------------------------------------


class _RawTriage:
    """A ClaudeTriage-style triage whose LLM returns arbitrary raw text (tests _json_call parsing)."""

    def __init__(self, raw: str):
        from synapse.core.write_pipeline import ClaudeTriage

        self._t = ClaudeTriage.__new__(ClaudeTriage)  # skip __init__ (no anthropic client needed)
        self._raw = raw

    async def classify(self, content, hint_type):
        return await _bound_classify(self._t, self._raw, content, hint_type)

    async def adjudicate(self, new_content, existing_fact):
        return await _bound_adjudicate(self._t, self._raw, new_content, existing_fact)


async def _bound_classify(triage, raw, content, hint_type):
    async def fake_call(system, user):
        return _lenient_parse(raw)
    triage._json_call = fake_call  # type: ignore[method-assign]
    from synapse.core.write_pipeline import ClaudeTriage
    return await ClaudeTriage.classify(triage, content, hint_type)


async def _bound_adjudicate(triage, raw, new_content, existing_fact):
    async def fake_call(system, user):
        return _lenient_parse(raw)
    triage._json_call = fake_call  # type: ignore[method-assign]
    from synapse.core.write_pipeline import ClaudeTriage
    return await ClaudeTriage.adjudicate(triage, new_content, existing_fact)


def _lenient_parse(raw: str):
    """Mimic _json_call's (data, parse_failed) contract without hitting an LLM."""
    import json as _json

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}, True
    try:
        return _json.loads(raw[start : end + 1]), False
    except ValueError:
        return {}, True


async def test_triage_parse_failure_fails_closed():
    # A weak local model emits broken JSON. The verdict MUST fail closed: worth_storing False,
    # parse_failed True — so the noise filter stays ON (R2), not silently disabled.
    triage = _RawTriage(raw="{ this is not valid json at all ")
    verdict = await triage.classify("Some content the model choked on.", None)
    assert verdict.worth_storing is False
    assert verdict.parse_failed is True

    # And the pipeline rejects it (unless forced).
    graphiti = FakeGraphiti()
    pipeline = WritePipeline(
        graphiti=graphiti, embedder=FakeEmbedder(), index=FakeIndex(None),
        triage=triage, dedup_threshold=0.9, relate_floor=0.75,
    )
    result = await pipeline.remember("Some content the model choked on.", project_id="acme-store")
    assert result.outcome is Outcome.REJECTED
    assert len(graphiti.calls) == 0


async def test_triage_valid_json_still_parsed():
    # Sanity: valid JSON is unaffected by the fail-closed change.
    triage = _RawTriage(raw='noise {"worth_storing": true, "knowledge_type": "lesson"} noise')
    verdict = await triage.classify("A durable lesson.", None)
    assert verdict.worth_storing is True
    assert verdict.parse_failed is False
    assert verdict.knowledge_type == "lesson"


# --- WP-B item 2: empty-extraction detection --------------------------------


class _NoEdgeResult:
    """add_episode result with zero fact edges (the local/degraded silent-failure signature)."""

    class _Ep:
        uuid = "ep-degraded"

    episode = _Ep()
    nodes = [_Node("SomeEntity")]
    edges: list = []


class _CountingDriver:
    def __init__(self):
        self.queries: list[str] = []

    async def execute_query(self, query, **params):
        self.queries.append(query)

        class _R:
            records: list = []

        return _R()


class FakeGraphitiNoEdges:
    def __init__(self, driver=None):
        self.calls: list[dict] = []
        if driver is not None:
            self.driver = driver

    async def add_episode(self, **kwargs):
        self.calls.append(kwargs)
        return _NoEdgeResult()


async def test_empty_extraction_flags_degraded():
    driver = _CountingDriver()
    graphiti = FakeGraphitiNoEdges(driver=driver)
    pipeline = WritePipeline(
        graphiti=graphiti, embedder=FakeEmbedder(), index=FakeIndex(None),
        triage=FakeTriage(STORABLE), dedup_threshold=0.9, relate_floor=0.75,
    )
    long_content = "A genuinely long and non-trivial decision statement " * 3  # > 80 chars
    result = await pipeline.remember(long_content, project_id="acme-store")

    assert result.outcome is Outcome.STORED       # still stored
    assert result.degraded is True                # but flagged
    assert result.facts_extracted == 0
    assert len(graphiti.calls) == 1
    # It queued a review entry (PendingCapture MERGE issued on the driver).
    assert any("PendingCapture" in q for q in driver.queries)


async def test_short_content_not_flagged_degraded():
    # Below the 80-char floor, zero edges is legitimate (a bare mention) — not flagged.
    graphiti = FakeGraphitiNoEdges(driver=_CountingDriver())
    pipeline = WritePipeline(
        graphiti=graphiti, embedder=FakeEmbedder(), index=FakeIndex(None),
        triage=FakeTriage(STORABLE), dedup_threshold=0.9, relate_floor=0.75,
    )
    result = await pipeline.remember("short note", project_id="acme-store")
    assert result.degraded is False
    assert result.facts_extracted == 0


async def test_facts_extracted_populated_on_normal_store():
    pipeline, _, _ = make_pipeline(verdict=STORABLE, nearest=None)
    result = await pipeline.remember("We chose SQLite for Acme-Store.", project_id="acme-store")
    assert result.facts_extracted == 1
    assert result.degraded is False


# --- WP-B item 4: content-hash exact-dup guard ------------------------------


class _HashHitDriver:
    """Driver whose Episodic-hash lookup returns a hit; records all queries issued."""

    def __init__(self, hit_uuid="ep-existing"):
        self.hit_uuid = hit_uuid
        self.queries: list[str] = []

    async def execute_query(self, query, **params):
        self.queries.append(query)

        records = []
        if "content_hash" in query and "MATCH" in query and "RETURN e.uuid" in query:
            records = [{"uuid": self.hit_uuid}]

        class _R:
            pass

        r = _R()
        r.records = records
        return r


class FakeGraphitiWithDriver:
    def __init__(self, driver):
        self.driver = driver
        self.calls: list[dict] = []

    async def add_episode(self, **kwargs):
        self.calls.append(kwargs)
        return _AddResult()


async def test_content_hash_dedup_exact_match():
    driver = _HashHitDriver(hit_uuid="ep-existing")
    graphiti = FakeGraphitiWithDriver(driver)
    pipeline = WritePipeline(
        graphiti=graphiti, embedder=FakeEmbedder(),
        index=FakeIndex(None),  # vector path would say "no nearest" — hash guard must win first
        triage=FakeTriage(STORABLE), dedup_threshold=0.9, relate_floor=0.75,
    )
    result = await pipeline.remember("Acme-Store uses SQLite.", project_id="acme-store")

    assert result.outcome is Outcome.DUPLICATE
    assert result.duplicate_of == "ep-existing"
    assert len(graphiti.calls) == 0   # never reached the store — deterministic short-circuit


def test_content_hash_normalizes_whitespace():
    # Cosmetic whitespace differences hash identically (so re-indents/newlines still dedup).
    assert _content_hash("a   b\n c") == _content_hash("a b c")
    assert _content_hash("  a b c  ") == _content_hash("a b c")
    assert _content_hash("a b c") != _content_hash("a b d")


# --- WP-B item 3: native vector index used ----------------------------------


class _VectorIndexDriver:
    """Records queries; returns a single record for the vector-index YIELD path."""

    def __init__(self):
        self.queries: list[str] = []

    async def execute_query(self, query, **params):
        self.queries.append(query)

        records = []
        if "queryRelationships" in query and "YIELD" in query:
            records = [{"uuid": "edge-vi", "fact": "X uses Y", "score": 0.83}]

        class _R:
            pass

        r = _R()
        r.records = records
        return r


class _Graphiti:
    def __init__(self, driver):
        self.driver = driver


async def test_vector_index_used_when_available():
    driver = _VectorIndexDriver()
    index = Neo4jVectorIndex(_Graphiti(driver))
    await index.ensure_index()
    nearest = await index.nearest([0.1] * 1024, ["project_acme-store"])

    # The nearest lookup issued the NATIVE relationship vector index call (not a brute-force scan).
    assert any("db.index.vector.queryRelationships" in q for q in driver.queries)
    # ensure_index created the vector index idempotently.
    assert any("CREATE VECTOR INDEX" in q and "IF NOT EXISTS" in q for q in driver.queries)
    assert nearest is not None
    assert nearest.uuid == "edge-vi"
    assert nearest.score == 0.83


async def test_vector_index_falls_back_to_scan_on_error():
    # If the index query raises, nearest() falls back to the brute-force cosine scan.
    class _RaisingThenScanDriver:
        def __init__(self):
            self.queries: list[str] = []

        async def execute_query(self, query, **params):
            self.queries.append(query)
            if "queryRelationships" in query:
                raise RuntimeError("index not online yet")

            class _R:
                records = [{"uuid": "edge-scan", "fact": "scanned", "score": 0.7}]

            return _R()

    driver = _RaisingThenScanDriver()
    index = Neo4jVectorIndex(_Graphiti(driver))
    index._index_ready = True  # skip ensure_index DDL
    nearest = await index.nearest([0.1] * 1024, ["project_acme-store"])

    assert any("vector.similarity.cosine" in q for q in driver.queries)  # scan fallback ran
    assert nearest is not None
    assert nearest.uuid == "edge-scan"


# --- redaction gate (research 2026-07-25 §5.1, Wave 0) -----------------------
# The contract is not just "the stored fact is clean" — it is that the credential never
# reaches ANY collaborator: not the triage LLM (an outbound API call), not the embedder
# (it would land in the vector index), not Graphiti. These fakes record what they were
# handed so each boundary is asserted independently.

FAKE_KEY = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8" * 4
SECRET_TEXT = f"Deploy uses ANTHROPIC_API_KEY={FAKE_KEY} and it must be rotated quarterly."


class RecordingEmbedder:
    def __init__(self):
        self.seen: list[str] = []

    async def create(self, input_data: str) -> list[float]:
        self.seen.append(input_data)
        return [0.1] * 1024


class RecordingTriage(FakeTriage):
    def __init__(self, verdict, adjudication=None):
        super().__init__(verdict, adjudication)
        self.classified: list[str] = []

    async def classify(self, content, hint_type):
        self.classified.append(content)
        return await super().classify(content, hint_type)


def make_recording_pipeline(*, verdict=STORABLE, nearest=None):
    graphiti = FakeGraphiti()
    embedder = RecordingEmbedder()
    triage = RecordingTriage(verdict)
    pipeline = WritePipeline(
        graphiti=graphiti, embedder=embedder, index=FakeIndex(nearest), triage=triage,
        dedup_threshold=0.9, relate_floor=0.75,
    )
    return pipeline, graphiti, embedder, triage


async def test_secret_never_reaches_triage_embedder_or_graph():
    pipeline, graphiti, embedder, triage = make_recording_pipeline()
    result = await pipeline.remember(SECRET_TEXT, project_id="acme-store")

    assert result.outcome is Outcome.STORED
    # 1. the triage LLM (an outbound Anthropic call) never saw the key
    assert triage.classified and all(FAKE_KEY not in c for c in triage.classified)
    # 2. the embedder never saw it → it cannot enter the vector index
    assert embedder.seen and all(FAKE_KEY not in s for s in embedder.seen)
    # 3. Graphiti never saw it → it cannot enter the graph
    assert all(FAKE_KEY not in call["episode_body"] for call in graphiti.calls)
    # 4. the caller is told what was stripped, by kind only
    assert "anthropic_api_key" in result.redactions
    assert not any(FAKE_KEY in r for r in result.redactions)


async def test_surrounding_knowledge_survives_redaction():
    pipeline, graphiti, _, _ = make_recording_pipeline()
    await pipeline.remember(SECRET_TEXT, project_id="acme-store")
    stored = graphiti.calls[0]["episode_body"]
    assert "must be rotated quarterly" in stored, "we redact the secret, not the lesson"
    assert "ANTHROPIC_API_KEY" in stored, "the key NAME is useful knowledge"


async def test_content_hash_is_computed_on_redacted_content():
    # Otherwise two writes differing only in their (stripped) secret would hash differently
    # and both be stored, defeating the exact-duplicate guard.
    pipeline, graphiti, _, _ = make_recording_pipeline()
    await pipeline.remember(SECRET_TEXT, project_id="acme-store")
    stored = graphiti.calls[0]["episode_body"]
    assert _content_hash(stored) == _content_hash(redact(SECRET_TEXT)[0])


async def test_clean_content_reports_no_redactions():
    pipeline, _, _, _ = make_recording_pipeline()
    result = await pipeline.remember("We chose SQLite for Acme-Store.", project_id="acme-store")
    assert result.redactions == []


async def test_redaction_reported_even_when_write_is_rejected():
    # A rejected write still altered content; the caller must learn a secret was present
    # (it means the agent is echoing credentials, regardless of storage outcome).
    rejected = TriageVerdict(worth_storing=False, knowledge_type="entity", reason="noise")
    pipeline, _, _, _ = make_recording_pipeline(verdict=rejected)
    result = await pipeline.remember(SECRET_TEXT, project_id="acme-store")
    assert result.outcome is Outcome.REJECTED
    assert "anthropic_api_key" in result.redactions


async def test_redaction_reported_on_the_duplicate_path():
    pipeline, _, _, _ = make_recording_pipeline(
        nearest=NearestFact(uuid="edge-1", fact="existing", score=0.95)
    )
    result = await pipeline.remember(SECRET_TEXT, project_id="acme-store")
    assert result.outcome is Outcome.DUPLICATE
    assert "anthropic_api_key" in result.redactions


# --- dedup scope override (roadmap item 23) -----------------------------------


def test_dedup_scopes_defaults_to_the_write_scope_only():
    pipeline, _, _, _ = make_recording_pipeline()
    assert pipeline._dedup_scopes("project_acme-store") == ["project_acme-store"]


def test_dedup_scopes_honours_an_explicit_override():
    pipeline, _, _, _ = make_recording_pipeline()
    assert pipeline._dedup_scopes("cluster_trading", ["cluster_trading", "global"]) == [
        "cluster_trading", "global",
    ]
    # an empty override is not an override
    assert pipeline._dedup_scopes("project_x", []) == ["project_x"]


class _RecordingIndex:
    def __init__(self, nearest=None):
        self._nearest = nearest
        self.scopes = None

    async def nearest(self, vec, scopes):
        self.scopes = scopes
        return self._nearest


async def test_widened_dedup_scopes_reach_the_vector_index():
    graphiti = FakeGraphiti()
    index = _RecordingIndex()
    pipeline = WritePipeline(graphiti=graphiti, embedder=FakeEmbedder(), index=index,
                            triage=FakeTriage(STORABLE), dedup_threshold=0.9, relate_floor=0.75)
    await pipeline.remember("domain knowledge", cluster="trading",
                            dedup_scopes=["cluster_trading", "global"])
    assert index.scopes == ["cluster_trading", "global"]


async def test_a_promotion_style_write_is_caught_as_a_duplicate_of_the_wider_tier():
    # The exact failure from the first real promotion: cluster_trading was empty, but global
    # already held the knowledge.
    graphiti = FakeGraphiti()
    index = _RecordingIndex(NearestFact(uuid="global-edge", fact="already known", score=0.99))
    pipeline = WritePipeline(graphiti=graphiti, embedder=FakeEmbedder(), index=index,
                            triage=FakeTriage(STORABLE), dedup_threshold=0.9, relate_floor=0.75)
    result = await pipeline.remember("already known", cluster="trading",
                                     dedup_scopes=["cluster_trading", "global"])
    assert result.outcome is Outcome.DUPLICATE
    assert result.duplicate_of == "global-edge"
    assert graphiti.calls == [], "nothing may be stored"


# --- global-write gate (research §5.3, roadmap item 15) ------------------------
# `global` is the one scope composed for EVERY project, and the UserPromptSubmit hook injects it
# into every prompt — so a project-specific fact there is noise eleven times over. Measured on the
# live graph: of 137 active global facts, 41 (30%) named exactly one project.

_PROJECT_CLUSTERS = {
    "acme-sim": "trading", "acme-api": "trading", "acme-data": "trading",
    "acme-flow": "infra", "acme-bot": "infra",
    "acme-docs": "creative", "acme-store": "creative",
    "loner": None,
}


def _gated_pipeline(**kw):
    graphiti = FakeGraphiti()
    return WritePipeline(
        graphiti=graphiti, embedder=FakeEmbedder(), index=FakeIndex(None),
        triage=FakeTriage(STORABLE), dedup_threshold=0.9, relate_floor=0.75,
        known_projects=lambda: list(_PROJECT_CLUSTERS),
        cluster_resolver=_PROJECT_CLUSTERS.get, **kw
    ), graphiti


def test_a_global_write_naming_one_project_is_refiled_to_that_project():
    pipeline, _ = _gated_pipeline()
    assert pipeline._better_scope_than_global(
        "The decision to use ib_async instead of ib_insync applies to the Acme-Sim project"
    ) == "project_acme-sim"


def test_a_global_write_naming_two_projects_in_one_cluster_is_refiled_to_the_cluster():
    # The measured real case: the SAME knowledge sat in global twice, once per trading project.
    pipeline, _ = _gated_pipeline()
    assert pipeline._better_scope_than_global(
        "The broker historical volume-in-lots gotcha applies to Acme-Sim and Acme-API alike"
    ) == "cluster_trading"


def test_a_write_spanning_clusters_stays_global():
    pipeline, _ = _gated_pipeline()
    assert pipeline._better_scope_than_global("acme-flow and acme-docs both run on Kafka") is None


def test_a_write_naming_no_project_stays_global():
    pipeline, _ = _gated_pipeline()
    assert pipeline._better_scope_than_global(
        "BigDecimal must be used for monetary values in Java, never double"
    ) is None


def test_a_project_without_a_cluster_cannot_be_refiled_to_one():
    pipeline, _ = _gated_pipeline()
    assert pipeline._better_scope_than_global("loner and acme-sim disagree") is None


def test_project_names_match_on_word_boundaries_only():
    # Regression for the actual bug that made this gate a silent no-op: the pattern was written as
    # rf"\b..." inside a NON-raw string, so \b became a literal backspace (0x08) and the regex
    # "\x08acme-sim\x08" never matched anything. The gate returned None for every input.
    pipeline, _ = _gated_pipeline()
    assert pipeline._better_scope_than_global("Acme-Store uses Firebase") == "project_acme-store"
    # ...and a name embedded inside a longer word must NOT count.
    assert pipeline._better_scope_than_global("the forgery detector flags fakes") is None


async def test_the_gate_refiles_a_real_write_and_reports_the_redirect():
    pipeline, graphiti = _gated_pipeline()
    result = await pipeline.remember(
        "The ib_async decision applies to the Acme-Sim project", project_id=None,
    )
    assert result.scope == "project_acme-sim"
    assert result.scope_redirected_from == "global"
    assert "global-write gate" in result.reason
    assert graphiti.calls[0]["group_id"] == "project_acme-sim"


async def test_an_ungated_global_write_is_untouched():
    pipeline, graphiti = _gated_pipeline()
    result = await pipeline.remember("BigDecimal for money in Java, never double", project_id=None)
    assert result.scope == "global"
    assert result.scope_redirected_from is None
    assert graphiti.calls[0]["group_id"] == "global"


async def test_consolidation_is_trusted_to_write_global_directly():
    # Promotion reaches global only through a reviewed, evidence-backed proposal — gating it would
    # fight the mechanism that exists to populate global correctly.
    pipeline, graphiti = _gated_pipeline()
    result = await pipeline.remember(
        "The broker gotcha applies to Acme-Sim and Acme-API", project_id=None, source="consolidation",
    )
    assert result.scope == "global"
    assert result.scope_redirected_from is None


async def test_an_explicit_project_write_never_reaches_the_gate():
    pipeline, graphiti = _gated_pipeline()
    result = await pipeline.remember("something about Acme-Sim", project_id="acme-api")
    assert result.scope == "project_acme-api"
    assert result.scope_redirected_from is None


def test_no_resolver_means_no_gating():
    pipeline, _, _, _ = make_recording_pipeline()
    assert pipeline._better_scope_than_global("applies to the Acme-Sim project") is None


def test_a_broken_registry_never_blocks_a_global_write():
    def boom():
        raise RuntimeError("registry unreadable")

    pipeline, _ = _gated_pipeline()
    pipeline._known_projects = boom
    assert pipeline._better_scope_than_global("applies to Acme-Sim") is None


# --- top-k contradiction adjudication (roadmap item 16) -----------------------
# Adjudicating only the SINGLE nearest fact meant a write contradicting the second-nearest was
# never flagged. Measured consequence: 7 Contradicts edges across 3,039 facts.


class TopKIndex:
    """Index that serves a ranked candidate list, like the real Neo4jVectorIndex."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.k_asked = None

    async def nearest(self, vec, scopes):
        return self._candidates[0] if self._candidates else None

    async def nearest_k(self, vec, scopes, k):
        self.k_asked = k
        return self._candidates[:k]


class ScriptedTriage(FakeTriage):
    """Returns a verdict per adjudicated fact, so ordering can be asserted."""

    def __init__(self, verdict, by_fact):
        super().__init__(verdict)
        self._by_fact = by_fact
        self.seen: list[str] = []

    async def adjudicate(self, new_content, existing_fact):
        self.seen.append(existing_fact)
        return Adjudication(relation=self._by_fact.get(existing_fact, Relation.DISTINCT))


def _topk_pipeline(candidates, by_fact, **kw):
    graphiti = FakeGraphiti()
    index = TopKIndex(candidates)
    triage = ScriptedTriage(STORABLE, by_fact)
    pipeline = WritePipeline(
        graphiti=graphiti, embedder=FakeEmbedder(), index=index, triage=triage,
        dedup_threshold=0.9, relate_floor=0.75, **kw
    )
    return pipeline, graphiti, index, triage


def _c(uuid, fact, score):
    return NearestFact(uuid=uuid, fact=fact, score=score)


async def test_a_contradiction_with_the_second_nearest_fact_is_now_flagged():
    # The whole point of the item: previously only candidate #1 was ever judged.
    candidates = [_c("n1", "unrelated but similar wording", 0.86),
                  _c("n2", "the opposite claim", 0.84)]
    pipeline, _g, _i, triage = _topk_pipeline(
        candidates, {"the opposite claim": Relation.CONTRADICTION})
    result = await pipeline.remember("the new claim", project_id="x")
    assert result.outcome is Outcome.CONTRADICTION
    assert result.contradicts == "n2"
    assert triage.seen == ["unrelated but similar wording", "the opposite claim"]


async def test_a_duplicate_anywhere_in_the_band_wins_over_a_contradiction():
    # Storing a duplicate is worse than missing one contradiction flag.
    candidates = [_c("n1", "contradicting fact", 0.88), _c("n2", "same thing restated", 0.80)]
    pipeline, graphiti, _i, _t = _topk_pipeline(candidates, {
        "contradicting fact": Relation.CONTRADICTION,
        "same thing restated": Relation.DUPLICATE,
    })
    result = await pipeline.remember("new content", project_id="x")
    assert result.outcome is Outcome.DUPLICATE
    assert result.duplicate_of == "n2"
    assert graphiti.calls == [], "a duplicate must not be stored"


async def test_the_highest_similarity_contradiction_is_the_one_reported():
    candidates = [_c("n1", "first contradiction", 0.88), _c("n2", "second contradiction", 0.80)]
    pipeline, _g, _i, _t = _topk_pipeline(candidates, {
        "first contradiction": Relation.CONTRADICTION,
        "second contradiction": Relation.CONTRADICTION,
    })
    result = await pipeline.remember("new content", project_id="x")
    assert result.contradicts == "n1"


async def test_adjudication_is_capped_because_each_one_costs_an_llm_call():
    candidates = [_c(f"n{i}", f"fact {i}", 0.85 - i * 0.01) for i in range(5)]
    pipeline, _g, _i, triage = _topk_pipeline(candidates, {}, max_adjudications=2)
    await pipeline.remember("new content", project_id="x")
    assert len(triage.seen) == 2


async def test_candidates_outside_the_gray_band_are_never_adjudicated():
    # Below relate_floor is unrelated; at/above dedup_threshold is settled deterministically.
    candidates = [_c("n1", "way below the floor", 0.40), _c("n2", "also below", 0.10)]
    pipeline, graphiti, _i, triage = _topk_pipeline(candidates, {})
    result = await pipeline.remember("new content", project_id="x")
    assert triage.seen == []
    assert result.outcome is Outcome.STORED and len(graphiti.calls) == 1


async def test_a_deterministic_duplicate_short_circuits_before_any_adjudication():
    candidates = [_c("n1", "near identical", 0.97), _c("n2", "other", 0.80)]
    pipeline, _g, _i, triage = _topk_pipeline(candidates, {})
    result = await pipeline.remember("new content", project_id="x")
    assert result.outcome is Outcome.DUPLICATE and result.duplicate_of == "n1"
    assert triage.seen == [], "no LLM call needed when cosine already settles it"


async def test_the_configured_candidate_width_is_requested():
    candidates = [_c("n1", "a", 0.80)]
    pipeline, _g, index, _t = _topk_pipeline(candidates, {}, candidate_k=7)
    await pipeline.remember("new content", project_id="x")
    assert index.k_asked == 7


async def test_an_index_without_topk_support_degrades_to_the_nearest_fact():
    # Older builds / unit fakes expose only nearest(); adjudication must still work on top-1.
    pipeline, _g, _e, triage = make_recording_pipeline(
        nearest=NearestFact(uuid="n1", fact="the opposite claim", score=0.85))
    pipeline.triage = ScriptedTriage(STORABLE, {"the opposite claim": Relation.CONTRADICTION})
    result = await pipeline.remember("new claim", project_id="x")
    assert result.outcome is Outcome.CONTRADICTION and result.contradicts == "n1"


async def test_a_topk_lookup_failure_falls_back_rather_than_failing_the_write():
    class Exploding(TopKIndex):
        async def nearest_k(self, vec, scopes, k):
            raise RuntimeError("index offline")

    graphiti = FakeGraphiti()
    index = Exploding([_c("n1", "fallback fact", 0.80)])
    pipeline = WritePipeline(graphiti=graphiti, embedder=FakeEmbedder(), index=index,
                            triage=FakeTriage(STORABLE), dedup_threshold=0.9, relate_floor=0.75)
    result = await pipeline.remember("new content", project_id="x")
    assert result.outcome is Outcome.STORED


async def test_a_contradiction_is_persisted_on_the_superseded_fact():
    # Before this the flag lived only in the write response, which is why the live graph held 7
    # Contradicts edges against 3,039 facts.
    class _Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **params):
            self.calls.append((query, params))

            class _R:
                records = []

            return _R()

    class _G:
        def __init__(self, d):
            self.driver = d

    driver = _Driver()
    pipeline = WritePipeline(graphiti=_G(driver), embedder=None, index=None, triage=None)
    await pipeline._persist_contradiction("ep-1", "old-edge")
    query, params = next((q, p) for q, p in driver.calls if "contradicted_by" in q)
    assert params["old"] == "old-edge" and params["ep"] == "ep-1"
    assert "contradicted_at" in query


async def test_persisting_a_contradiction_never_undoes_a_successful_write():
    class _Boom:
        async def execute_query(self, query, **params):
            raise RuntimeError("neo4j down")

    class _G:
        driver = _Boom()

    pipeline = WritePipeline(graphiti=_G(), embedder=None, index=None, triage=None)
    await pipeline._persist_contradiction("ep-1", "old-edge")  # must not raise
