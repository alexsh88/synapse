"""Unit tests for the write pipeline (plan Part 4).

All external services (Claude triage, BGE-M3 embedder, Neo4j vector index, the
Graphiti graph) are faked, so these test the pipeline's *logic* — the write
trigger, scope resolution, dedup threshold, and contradiction flagging — without
touching the network. A live end-to-end run lives in scripts/schema_smoke.py.
"""

from __future__ import annotations

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
