"""Synapse retrieval engine (plan Part 5). How knowledge comes out.

Multi-strategy retrieval, a tunable ranking algorithm, scope composition, and the
``brief()`` killer feature with Redis caching.

**Semantic search runs over Neo4j's vector index via Graphiti** (`graphiti.search`
already combines cosine similarity + BM25 + graph BFS), not a separate Qdrant —
the write pipeline stores embeddings through Graphiti into Neo4j, so that is where
the vectors live. This confirms the §6C question: Qdrant is currently redundant.
See docs/architecture/retrieval.md.

All graph access is behind injected Protocols (`Searcher`, `GraphQueries`) so the
ranking, temporal filtering, scope composition, and brief assembly are unit-tested
without live services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, Field

from synapse.config import settings
from synapse.core import query_log
from synapse.core.schema import Scope

logger = logging.getLogger(__name__)

# Knowledge-type labels used to bucket a brief.
_CONVENTION, _DECISION, _LESSON, _PATTERN = "Convention", "Decision", "Lesson", "Pattern"

# BGE-M3 assigns a high baseline cosine (~0.65) even to unrelated text, so raw cosine is not a
# usable [0,1] relevance on its own — 0.65 means "nothing in common". Relevance is rescaled from
# this baseline (or from the engine's similarity floor when one is set). See _relevance_scores.
COSINE_BASELINE = 0.65

# Hard bottom of the rescue band (see apply_similarity_floor). Sits below the lowest cosine at
# which the live corpus was measured to still return a correct answer (0.678) and above the point
# where "shares a rare term" stops being evidence of anything.
RESCUE_FLOOR = 0.66

# A query term appearing in more than this fraction of the candidate pool is not discriminative
# enough to rescue a below-floor fact. 0.25 keeps "ensemble" (3/59) and rejects "service" (~30/60).
_ANCHOR_MAX_DF = 0.25

# Widest candidate pool measured to keep scope isolation intact. At 12x the eval recorded its first
# violation (a trading fact surfacing in a creative project), so this is a SAFETY ceiling, not a
# performance one — see MEASURED_SAFE_MULTIPLIER's use in RetrievalEngine.__init__.
MEASURED_SAFE_MULTIPLIER = 6

# RRF damping constant (see rrf_fuse). NOT the canonical 60: that value is calibrated for merging
# result lists of thousands, where rank 1 and rank 5 should score almost identically. Synapse fuses
# pools of ~30-60 candidates, and at k=60 the entire pool spans 1/61..1/120 — every rank difference
# collapses into noise and fusion cannot outvote a miscalibrated score. 10 keeps real separation
# across the top of a small pool.
_RRF_K = 10

# The schema models node confidence as a categorical Literal, not a float
# (Decision.confidence in synapse/core/schema.py). Live distribution 2026-07-25:
# settled 142, tentative 15, locked 8.
CONFIDENCE_WORDS: dict[str, float] = {"tentative": 0.4, "settled": 0.75, "locked": 1.0}

# Node attributes surfaced in briefs beyond name/summary. `alternatives_considered` and
# `chosen_over` are the highest-value text in the graph for an agent — they record what was
# already tried and rejected — and no read path selected them before 2026-07-25 (research §2.1).
_EXTRA_NODE_ATTRS = ("rationale", "alternatives_considered", "chosen_over")
_EXTRA_ATTR_MAXLEN = 180

# How many raw facts a brief carries, and how long each may be. The brief is injected at EVERY
# session start (~13KB already), so this section has to earn its context budget — 7 lines is the
# same allowance every other brief section gets.
_BRIEF_FACTS = 7
_BRIEF_FACT_MAXLEN = 200
# Token-set Jaccard at or above this counts two brief lines as the same TOPIC. Calibrated on the
# first live run of the facts section — see novel_lines() for the measured separation.
_BRIEF_OVERLAP = 0.18
# Candidate pool multiplier. Rows are short and the query is one indexed ORDER BY, so a wide pool
# is cheap; the topic filter is what decides the final count.
_BRIEF_FACT_OVERFETCH = 12

# Runbooks carry their full step list into the brief, so this section costs far more context per
# item than the one-line sections. 5 procedures is already a large budget.
_BRIEF_RUNBOOKS = 5


# ──────────────────────────────────────────────────────────────────────────────
# Normalized data shapes (decoupled from Graphiti types → easy to fake)
# ──────────────────────────────────────────────────────────────────────────────


class Fact(BaseModel):
    uuid: str
    fact: str
    group_id: str
    created_at: datetime | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    source_uuid: str | None = None
    target_uuid: str | None = None
    confidence: float | None = None
    score: float | None = None  # raw similarity from Graphiti/Neo4j (None if unavailable)
    # The fact's own 1024-dim vector, when the searcher could fetch it. Used ONLY for MMR
    # diversity (see :func:`mmr_rerank`); ``None`` degrades MMR to plain relevance order.
    embedding: list[float] | None = None


class NodeRow(BaseModel):
    uuid: str
    name: str
    summary: str | None = None
    labels: list[str] = Field(default_factory=list)
    group_id: str
    created_at: datetime | None = None
    attributes: dict = Field(default_factory=dict)


class RankWeights(BaseModel):
    """Tunable ranking weights.

    ``relevance`` / ``recency`` / ``confidence`` are ADDITIVE and normalized by their sum.
    ``connectivity`` is different: it is a **multiplicative bonus factor**, not a term in the
    sum (see :func:`score_facts`).

    Recalibrated 2026-07-25 after measuring the live corpus (research §1.1). The previous
    0.45/0.20/0.20/0.15 split gave 40% of the weight to two signals that carried almost no
    information: ``confidence`` was *never persisted on any edge* (all 3,011 edges resolved to
    the 0.5 default) and ``recency`` sat at ~0.286 for nearly every fact because the corpus was
    ingested inside one ~54-day window. Relevance therefore decided only ~56% of the ranking
    while a query-independent popularity prior decided ~19%.

    Split again 2026-07-25 (roadmap item 17): ``relevance`` gave up half its share to ``fusion``.
    The reason is not tuning but a measurement — absolute cosine is not a trustworthy relevance
    estimate on this corpus (correct facts span 0.670–0.865, junk spans 0.665–0.775, and for some
    queries the wrong answer outscores the right one), whereas the *agreement* of several rankings
    is. An even split is the "trust neither signal alone" position. Swept 0.0 → 0.70 fusion:
    MRR 0.699 → 0.721 and precision 0.466 → 0.530 monotonically, but ``diversity@k`` starts
    eroding past 0.35 (0.936 → 0.915 at 0.50) as fusion concentrates on one lexical cluster, and
    MRR is already flat by 0.35. Beyond that the gain is precision bought with diversity.

    These values are a measured starting point, not a truth: re-baseline with
    ``python -m scripts.run_eval`` after any change.
    """

    relevance: float = 0.35
    recency: float = 0.20
    confidence: float = 0.10
    # ADDITIVE, alongside relevance. Reciprocal-rank agreement across the retrieval lenses
    # (see :func:`rrf_fuse`). Exists because absolute cosine was measured to be an unreliable
    # relevance estimate — locally even anti-correlated — while the lenses' *rank* agreement is
    # not. Set to 0.0 to disable fusion entirely (the lenses then aren't even computed).
    fusion: float = 0.35
    # Multiplicative: composite = base * (1 + connectivity * conn). 0.10 => a maximally
    # connected fact gets at most a 10% lift, enough to break near-ties, never enough to
    # promote an irrelevant hub above a genuinely relevant fact.
    connectivity: float = 0.10
    recency_half_life_days: float = 30.0


class Recalled(BaseModel):
    fact: str
    score: float
    scope: str
    uuid: str
    valid_at: datetime | None = None
    components: dict[str, float] = Field(default_factory=dict)


class BriefRunbook(BaseModel):
    """A procedure in a brief, with its steps intact."""

    name: str
    scope: str
    steps: list[str]
    purpose: str | None = None
    prerequisites: str | None = None
    verified_at: datetime | None = None
    stale: bool = False


class Brief(BaseModel):
    project_id: str
    project_summary: str
    active_conventions: list[str]
    key_decisions: list[str]
    relevant_lessons: list[str]
    cross_project_knowledge: list[str]
    generated_at: datetime
    cached: bool = False
    # Recent fact edges that the label-driven sections above cannot reach (research §2.3
    # correction). Defaulted so briefs cached before this field existed still deserialize.
    recent_facts: list[str] = Field(default_factory=list)
    # Procedural memory (roadmap item 18). NOT flattened into `list[str]` like the sections above:
    # a runbook's steps are ordered and the order is the payload, so the brief carries the
    # structure rather than a one-line summary of it. A stale runbook is still shown — marked, not
    # hidden, because "the deploy sequence exists but nobody has verified it since April" is
    # exactly what the agent needs to know before following it.
    runbooks: list[BriefRunbook] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Injected collaborators
# ──────────────────────────────────────────────────────────────────────────────


class Searcher(Protocol):
    # scopes=None means "every scope" — GraphitiSearcher passes it straight through to
    # Graphiti's group_ids, which treats None as unfiltered. The Protocol said list[str],
    # which made the MCP `search` tool's cross-project path a type error against its own
    # implementation.
    async def search(
        self, query: str, scopes: list[str] | None, limit: int, center_node_uuid: str | None
    ) -> list[Fact]: ...


class GraphQueries(Protocol):
    async def nodes_by_label(self, labels: list[str], scopes: list[str], limit: int) -> list[NodeRow]: ...
    async def degrees(self, node_uuids: list[str]) -> dict[str, int]: ...
    async def node_confidence(self, node_uuids: list[str]) -> dict[str, float]: ...
    async def recent_facts(self, scopes: list[str], limit: int) -> list[Fact]: ...
    async def record_recall(self, fact_uuids: list[str]) -> None: ...
    # Optional: read via getattr, so a fake/older collaborator without it degrades to no
    # runbook section rather than breaking every brief.
    async def runbooks(self, scopes: list[str], limit: int): ...


# ──────────────────────────────────────────────────────────────────────────────
# Pure functions (the ranking + temporal logic — directly unit-tested)
# ──────────────────────────────────────────────────────────────────────────────


def temporal_filter(facts: list[Fact], as_of: datetime | None) -> list[Fact]:
    """Keep only facts valid at ``as_of`` (default: now).

    A fact is valid at T when ``valid_at <= T`` and (``invalid_at`` is None or
    ``invalid_at > T``). Facts with no temporal bounds are always kept.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    kept = []
    for f in facts:
        if f.valid_at is not None and f.valid_at > as_of:
            continue
        if f.invalid_at is not None and f.invalid_at <= as_of:
            continue
        kept.append(f)
    return kept


# Anchor extraction has its own stopword set, deliberately NOT `_LINE_STOPWORDS`. That set is
# tuned for brief topic-coverage and its docstring warns against sharing a tunable across
# purposes; widening it here to drop question-form words would silently retune brief filtering.
# These are the words a *question* is made of — they carry no topic signal even when rare.
_ANCHOR_STOPWORDS = frozenset(
    "best worst better worse good bad more most less least many much few new old "
    "use used uses using need needs needed want make makes made get gets got "
    "should would could does happens work works working way ways thing things "
    "recommended recommend required require handle handled handling supposed "
    "core main primary correct proper right wrong".split()
)


def discriminative_terms(
    facts: list[Fact], query: str, *, max_df: float = _ANCHOR_MAX_DF
) -> frozenset[str]:
    """Query terms rare enough *within this candidate pool* to anchor a low-similarity rescue.

    A poor-man's IDF computed over the candidates we already have in hand — no extra query, no
    corpus statistics to maintain, and self-calibrating per project. The point is to separate a
    term that actually discriminates ("ensemble" in 3 of 59 acme-jobs candidates) from one that
    merely appears ("service" in half of them), because only the former is evidence that a
    below-floor fact is on topic.

    Terms absent from every candidate (``df == 0``) are excluded too: they anchor nothing, and an
    off-topic query is mostly made of them ("sourdough", "hypertension").
    """
    tokens = _line_tokens(query) - _ANCHOR_STOPWORDS
    if not tokens or not facts:
        return frozenset()
    cutoff = max(1, int(len(facts) * max_df))
    counts = dict.fromkeys(tokens, 0)
    for f in facts:
        for term in tokens & _line_tokens(f.fact):
            counts[term] += 1
    return frozenset(t for t, df in counts.items() if 0 < df <= cutoff)


def apply_similarity_floor(
    facts: list[Fact],
    min_relevance: float | None,
    *,
    query: str | None = None,
    rescue_floor: float | None = None,
) -> list[Fact]:
    """Drop candidates whose absolute similarity ``score`` is below ``min_relevance``.

    **Rescue band (added 2026-07-25, roadmap item 17).** A single absolute threshold provably
    cannot do this job. Measured over the 52-case golden set on the live graph: correct facts span
    cosine 0.670–0.865 while the top candidate of an off-topic/leakage query spans 0.665–0.775 —
    **six of forty positives score below the best negative**. At 0.72 the floor was silently
    deleting three known-correct answers (``LightGBM + XGBoost soft-voting ensemble`` at 0.7198,
    two thousandths short) to reject junk it was letting through anyway.

    So below the floor we stop asking "is the similarity high enough" and ask a different question:
    does the fact share a *discriminative* term with the query (see :func:`discriminative_terms`)?
    A fact between ``rescue_floor`` and ``min_relevance`` survives only on that lexical evidence.
    ``rescue_floor`` remains a hard bottom so a shared term cannot drag in genuinely distant facts.

    Without ``query``/``rescue_floor`` this is the plain single-threshold filter it always was.

    * ``min_relevance is None`` or ``<= 0`` → no-op (floor disabled).
    * A fact with ``score is None`` (no absolute similarity available — e.g. the
      searcher couldn't compute cosine) is KEPT: we never had a signal to reject
      it on, and silently dropping everything would be worse than a soft miss.

    The floor still guards against *confident junk*: BGE-M3 gives a high baseline cosine (~0.65)
    to unrelated text, so an unfiltered off-topic query returns a full, plausible-looking result
    set. It is applied to the *raw* similarity, not the rescaled relevance — the latter is derived
    from this threshold, so gating on it would be circular.
    """
    if not min_relevance or min_relevance <= 0:
        return facts
    anchors: frozenset[str] = frozenset()
    if query and rescue_floor is not None:
        anchors = discriminative_terms(facts, query)
    kept: list[Fact] = []
    for f in facts:
        if f.score is None or f.score >= min_relevance:
            kept.append(f)
        elif anchors and rescue_floor is not None and f.score >= rescue_floor:
            if _line_tokens(f.fact) & anchors:
                kept.append(f)
    return kept


# Function words must not count toward line overlap. Without this, two unrelated short sentences
# share "the"/"for"/"and" and read as ~0.18 similar — which collapsed two genuinely distinct Redis
# facts on the first attempt. Only content words carry topic signal.
_LINE_STOPWORDS = frozenset(
    "the and for are was were its his her their our your with from into onto over under "
    "this that these those but not nor yet all any both each some such only just also still "
    "even other another same than then when while because since although though whether "
    "which who whom whose what where how now here there already always never ever one "
    "has have had having will would shall should can could may might must does did done "
    "been being via per about across without within".split()
)


def _line_tokens(text: str) -> frozenset[str]:
    return frozenset(
        w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in _LINE_STOPWORDS
    )


def novel_lines(
    candidates: list[str], existing: list[str], limit: int, *, overlap: float = _BRIEF_OVERLAP,
) -> list[str]:
    """Pick up to *limit* candidate lines that say something the *existing* lines do not.

    Token-set Jaccard at or above ``overlap`` counts a candidate as already covered. Also filters
    candidates against each other, so the result is internally diverse too.

    ``overlap`` is tuned for TOPIC coverage, not for near-duplicate detection. The first live run
    of this section filled 6 of its 7 slots with one topic (the Bouchaud square-root impact law)
    at the old 0.5 default.

    Measured on those real lines, the two populations are **not** cleanly separable pairwise:
    same-topic pairs span 0.045–0.800 and different-topic pairs span 0.000–0.091. A pairwise
    threshold alone would therefore be unreliable. What makes 0.18 work is that filtering is
    **greedy against every already-accepted line**, not pairwise: the topic's most central line is
    accepted first and collides with the rest of its cluster, so the low-scoring outliers
    (`"...uses a linear impact term"` vs `"...proportional to sqrt(order size)"`, only 0.045 to each
    other) are still caught via the line they both overlap. Widening the window is safe here for
    the same reason the risk profile below is asymmetric.

    Note the risk profile is the OPPOSITE of the consolidation engine's merge gate: dropping a line
    from a brief costs almost nothing (``recall`` still finds the fact), whereas topic flooding
    wastes the whole section. So this errs toward filtering, where the merge gate errs toward
    keeping.

    Deliberately a third, small implementation of "are these two lines the same idea": the eval
    harness has one as a *metric* (``_distinct_ideas``) and the consolidation engine has one as a
    *safety gate* (``content_difference``). Sharing a single tunable across measurement, a
    destructive-merge gate and cosmetic brief filtering would couple three things that must be
    tuned independently — a change to brief tidiness must never move a merge-safety threshold.
    """
    if limit <= 0:
        return []
    seen = [_line_tokens(e) for e in existing if e]
    picked: list[str] = []
    for candidate in candidates:
        text = _trim(candidate, _BRIEF_FACT_MAXLEN)
        if not text:
            continue
        tokens = _line_tokens(text)
        if not tokens:
            continue
        if any(
            len(tokens & other) / len(tokens | other) >= overlap
            for other in seen
            if other
        ):
            continue
        picked.append(text)
        seen.append(tokens)
        if len(picked) >= limit:
            break
    return picked


def _trim(value: object, limit: int = _EXTRA_ATTR_MAXLEN) -> str:
    """Normalize a node attribute to a single trimmed line, or '' when absent/blank.

    Briefs are injected into every session, so an unbounded rationale would tax the context
    budget it exists to save (T2/T11).
    """
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _recency_score(when: datetime | None, now: datetime, half_life_days: float) -> float:
    if when is None:
        return 0.5
    age_days = max(0.0, (now - when).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def _relevance_scores(facts: list[Fact], baseline: float = COSINE_BASELINE) -> list[float]:
    """Per-fact relevance in [0, 1], rescaled from an ABSOLUTE similarity baseline.

    Changed 2026-07-25 (research §1.4). This used to min-max normalize within the result set,
    which had two costs:

    * The best hit was **always 1.0**, even when it was mediocre — an off-topic query that
      squeaked past the floor still presented a confident-looking ``relevance=1.000``.
    * Scores were incomparable across queries, so no caller could threshold on them. That is
      exactly why ``scripts/prompt_recall.py`` cannot re-gate on the returned score.

    Rescaling from a fixed baseline — ``(s - baseline) / (1 - baseline)``, clamped — makes
    relevance absolute and comparable across queries. ``baseline`` should be the engine's
    similarity floor when one is set, else :data:`COSINE_BASELINE`.

    A fact with ``score is None`` gets a neutral 0.5 (we never had a signal to judge it on).
    If NO fact carries a score, fall back to positional ``1 - i/n`` from the searcher's order.
    """
    scores = [f.score for f in facts]
    present = [s for s in scores if s is not None]
    if present:
        span = max(1.0 - baseline, 1e-9)
        return [
            0.5 if s is None else min(1.0, max(0.0, (s - baseline) / span))
            for s in scores
        ]
    n = max(len(facts), 1)
    return [1.0 - (i / n) for i in range(len(facts))]


def lens_ranks(facts: list[Fact], query: str) -> dict[str, list[str]]:
    """Rank the same candidates three ways — the retrieval "lenses" of roadmap item 17.

    Each lens returns fact uuids best-first. They disagree, which is the entire point: fusing
    disagreeing rankings is robust to any single lens being miscalibrated, and absolute cosine was
    measured to be exactly that.

    * ``hybrid``  — the order the searcher already returned. Graphiti ranks with BM25 + vector +
      its own reranker, and this code **used to throw that ordering away** and re-sort by raw
      cosine. On "what carries events between services?" Graphiti had the correct Kafka facts at
      #1/#3 while cosine put an irrelevant `bearer-token is forwarded by service` above them. The
      most valuable lens was already being computed and discarded; it costs nothing to keep.
    * ``cosine``  — absolute similarity, descending. The previous sole signal.
    * ``lexical`` — how many *discriminative* query terms the fact contains (see
      :func:`discriminative_terms`), descending, cosine breaking ties. A crude BM25 stand-in that
      answers the one question cosine cannot: does this fact mention what was asked about?

    No lens issues a query — all three re-rank candidates already in hand.
    """
    hybrid = [f.uuid for f in facts]
    cosine = [f.uuid for f in sorted(facts, key=lambda f: (f.score is not None, f.score or 0.0), reverse=True)]
    anchors = discriminative_terms(facts, query)
    lexical = [
        f.uuid
        for f in sorted(
            facts,
            key=lambda f: (len(_line_tokens(f.fact) & anchors), f.score or 0.0),
            reverse=True,
        )
    ]
    return {"hybrid": hybrid, "cosine": cosine, "lexical": lexical}


def rrf_fuse(
    lenses: dict[str, list[str]], *, k: int = _RRF_K, weights: dict[str, float] | None = None
) -> dict[str, float]:
    """Reciprocal Rank Fusion over per-lens rankings, normalized to [0, 1].

    ``score(d) = Σ_lens w_lens / (k + rank_lens(d))``, then divided by the best score in the set
    so the winner is 1.0 and the value is usable as a ranking component.

    ``k`` damps the influence of top ranks. The canonical RRF constant is 60, chosen for result
    lists of thousands; against candidate pools of ~60 it flattens every rank difference to noise.
    See :data:`_RRF_K` for the value used here and why.

    A document missing from a lens contributes nothing from that lens, which is the desired
    behaviour: absence is weak evidence against, not a hard veto.
    """
    if not lenses:
        return {}
    weights = weights or {}
    raw: dict[str, float] = {}
    for name, ranking in lenses.items():
        w = weights.get(name, 1.0)
        for rank, uuid in enumerate(ranking, start=1):
            raw[uuid] = raw.get(uuid, 0.0) + w / (k + rank)
    best = max(raw.values(), default=0.0)
    if best <= 0.0:
        return dict.fromkeys(raw, 0.0)
    return {uuid: score / best for uuid, score in raw.items()}


def _confidence_for(fact: Fact, node_confidence: dict[str, float]) -> float:
    """Resolve a fact's confidence in [0, 1].

    Order: the edge's own float ``confidence`` → the best categorical confidence of its endpoint
    nodes (``CONFIDENCE_WORDS``) → 0.5.

    The middle step exists because the corpus measured 2026-07-25 had confidence on **zero of
    3,011 edges** but on **165 of 167 Decision nodes** — a real signal the ranker was ignoring
    entirely while still spending 20% of its weight on it (research §1.1). Endpoints are combined
    with ``max``: a fact touching a *locked* decision is at least as trustworthy as that decision.
    """
    if fact.confidence is not None:
        return fact.confidence
    values = [
        node_confidence[u]
        for u in (fact.source_uuid, fact.target_uuid)
        if u and u in node_confidence
    ]
    return max(values) if values else 0.5


def score_facts(
    facts: list[Fact],
    weights: RankWeights,
    connectivity: dict[str, float],
    now: datetime | None = None,
    *,
    node_confidence: dict[str, float] | None = None,
    relevance_baseline: float = COSINE_BASELINE,
    fusion: dict[str, float] | None = None,
) -> list[tuple[Fact, float, dict[str, float]]]:
    """Combine relevance + recency + confidence, then apply a connectivity bonus.

    Returns ``(fact, composite, components)`` sorted by descending composite.

    **Connectivity is multiplicative, not additive** (changed 2026-07-25, research §1.2). As an
    additive term it was a query-independent popularity prior that let hub facts free-ride into
    every result set: measured on the live graph, a fact with ``relevance=0.027`` — essentially
    unrelated to the query — still scored 0.323 purely on ``connectivity=1.0``, outranking facts
    with 15x its relevance. With 38 nodes at degree 11+ and 838 at degree <=1, a handful of hubs
    were positioned to contaminate everything.

    As a bounded multiplier it can only reorder near-ties::

        composite = (w_rel*rel + w_rec*rec + w_conf*conf) / (w_rel+w_rec+w_conf)
                    * (1 + w_conn * conn)

    The composite is clamped to 1.0 so it stays a [0,1] score.
    """
    now = now or datetime.now(timezone.utc)
    node_confidence = node_confidence or {}
    fusion = fusion or {}
    # `fusion` joins the additive sum only when it carries weight, so a caller that passes no
    # lenses (or leaves weights.fusion at 0) gets bit-identical scores to before.
    fusion_w = weights.fusion if fusion else 0.0
    additive_w = (weights.relevance + weights.recency + weights.confidence + fusion_w) or 1.0
    relevances = _relevance_scores(facts, relevance_baseline)
    scored = []
    for i, f in enumerate(facts):
        relevance = relevances[i]
        recency = _recency_score(f.valid_at or f.created_at, now, weights.recency_half_life_days)
        confidence = _confidence_for(f, node_confidence)
        fused = fusion.get(f.uuid, 0.0)
        # Facts with no degree data (manual relate() edges, endpoints outside the batch) get 0.0,
        # not a neutral 0.5 — an unknown connection count must not outrank a measured one.
        conn = connectivity.get(f.uuid, 0.0)
        base = (
            weights.relevance * relevance
            + weights.recency * recency
            + weights.confidence * confidence
            + fusion_w * fused
        ) / additive_w
        composite = min(1.0, base * (1.0 + weights.connectivity * conn))
        components = {
            "relevance": relevance, "recency": recency,
            "confidence": confidence, "connectivity": conn,
        }
        if fusion_w:
            components["fusion"] = fused
        scored.append((f, composite, components))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 when either is degenerate."""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _breaches_source_cap(fact: Fact, counts: dict[str, int], cap: int | None) -> bool:
    """True when this fact's source node already holds ``cap`` of the selected slots.

    Facts with no ``source_uuid`` are never capped — an unknown origin is not evidence of a
    monopoly, and rejecting on missing data would silently thin results.
    """
    if cap is None or fact.source_uuid is None:
        return False
    return counts.get(fact.source_uuid, 0) >= cap


def mmr_rerank(
    scored: list[tuple[Fact, float, dict[str, float]]],
    limit: int,
    *,
    lambda_: float = 0.7,
    max_per_source: int | None = None,
) -> list[tuple[Fact, float, dict[str, float]]]:
    """Re-order by Maximal Marginal Relevance so near-duplicate facts don't crowd the result.

    Added 2026-07-25 (research §1.3). Measured on the live graph, ``recall("how do we handle
    monetary values in java")`` returned **four restatements of "use BigDecimal for money" in six
    slots** — the 77 outstanding duplicate clusters surfacing at read time. That matters more than
    it looks: ``scripts/prompt_recall.py`` injects only 5 facts per prompt, so an agent could burn
    its entire injected context on one idea and never see the other things it needed.

    At each step pick the candidate maximizing::

        lambda_ * composite - (1 - lambda_) * max_similarity_to_already_selected

    ``lambda_=1.0`` is pure relevance (MMR off); lower values buy diversity at the cost of
    relevance. The top-ranked fact is always kept.

    Facts whose ``embedding`` is missing are never penalized (redundancy 0.0), so a searcher that
    couldn't fetch vectors degrades to plain relevance order rather than to arbitrary output.

    **``max_per_source`` — the hub-monopoly cap** (added 2026-07-25, roadmap item 17). Text
    similarity is not the only way a result set collapses. ``recall("what are the core ICT concepts
    the analyzer computes?")`` returned five facts that all hang off the same ``Acme Data``
    node — "built using React", "built using TypeScript", "uses a Node bridge server", "built for
    trading MES futures" — burying the answers that actually described the methodology. MMR
    happily kept all five because their TEXT is genuinely diverse; the redundancy is *structural*,
    an entity-attribute fan-out from one hub, and only the graph shape reveals it.

    The cap is soft: capped candidates are deferred, not deleted, and are taken anyway if the
    result set would otherwise come back short. Returning fewer facts than asked for would be a
    worse failure than an over-represented source.

    That softness makes the cap **useless on a narrow candidate pool**, which is worth stating
    because it is how the fix was found. At ``candidate_multiplier=3`` a cap of 2 still returned
    three facts from the same hub: only two source nodes were represented in the pool at all, so
    the cap exhausted its options and relaxed. It only starts working once the pool is wide enough
    to offer alternatives — measured on the live graph, ``max_per_source=2`` moved hit@k 0.769 →
    0.788 at ``candidate_multiplier=6`` and did nothing whatsoever at 3.
    """
    if limit <= 0 or not scored:
        return []
    if lambda_ >= 1.0 and max_per_source is None:
        return scored[:limit]

    remaining = list(scored)
    selected = [remaining.pop(0)]  # `scored` is sorted desc — the best fact is never dropped
    per_source: dict[str, int] = {}
    if selected[0][0].source_uuid:
        per_source[selected[0][0].source_uuid] = 1
    while remaining and len(selected) < limit:
        # Prefer candidates within the cap; fall back to the full pool rather than return short.
        pool = [
            i for i, (f, _s, _c) in enumerate(remaining)
            if not _breaches_source_cap(f, per_source, max_per_source)
        ]
        if not pool:
            pool = list(range(len(remaining)))
        best_idx, best_value = pool[0], float("-inf")
        for i in pool:
            f, composite, _c = remaining[i]
            redundancy = 0.0
            if f.embedding:
                sims = [
                    _cosine(f.embedding, chosen.embedding)
                    for chosen, _s, _cc in selected
                    if chosen.embedding
                ]
                redundancy = max(sims) if sims else 0.0
            value = lambda_ * composite - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_value, best_idx = value, i
        chosen_fact = remaining[best_idx][0]
        if chosen_fact.source_uuid:
            per_source[chosen_fact.source_uuid] = per_source.get(chosen_fact.source_uuid, 0) + 1
        selected.append(remaining.pop(best_idx))
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# The engine
# ──────────────────────────────────────────────────────────────────────────────


class RetrievalEngine:
    def __init__(
        self,
        searcher: Searcher,
        queries: GraphQueries,
        *,
        redis=None,
        weights: RankWeights | None = None,
        brief_ttl_seconds: int = 1800,
        candidate_multiplier: int = 6,
        min_relevance: float = 0.72,
        rescue_floor: float | None = RESCUE_FLOOR,
        mmr_lambda: float = 0.7,
        max_per_source: int | None = 2,
        cluster_resolver=None,
        log_queries: bool = True,
    ) -> None:
        self.searcher = searcher
        self.queries = queries
        self.redis = redis
        # Whether reads are recorded to the query log. On by default — the log is what a held-out
        # eval set is later mined from, and a query that was not recorded cannot be recovered.
        # Turned OFF by machinery that generates its own traffic: the eval harness, and the
        # pooling engines in scripts/build_heldout_set.py. Same reasoning as opt-in impressions
        # below — if the harness's own queries entered the log, the held-out set would be built
        # from the tuned set it exists to be independent of.
        self.log_queries = log_queries
        # project_id -> cluster name (or None). Injected rather than imported so the engine stays
        # registry-agnostic and unit-testable; wired to synapse.core.registry.cluster_of in
        # build_retrieval_engine. When absent, recall composes global + project as before.
        self.cluster_resolver = cluster_resolver
        self.weights = weights or RankWeights()
        self.brief_ttl = brief_ttl_seconds
        # Candidates fetched per requested result. Raised 3 -> 6 on 2026-07-25 (item 17) after the
        # measurement that mattered most: every ranking pass below can only re-order what it is
        # GIVEN, and at 3x a limit of 5 fetched 15 candidates while the correct answers for
        # `ict-methodology` sat at hybrid rank 19 and 25. No amount of fusion, diversity or
        # capping could reach them. Widening is not free in either direction — at 12x the pool
        # starts admitting cross-project leakage (a measured violation, see mmr_rerank).
        self.candidate_multiplier = candidate_multiplier
        # MMR diversity trade-off (1.0 = off/pure relevance). See :func:`mmr_rerank`.
        self.mmr_lambda = mmr_lambda
        # Max results allowed to share one source node — the hub-monopoly cap. See mmr_rerank.
        self.max_per_source = max_per_source
        # Absolute cosine floor. Off-topic queries hover ~0.65–0.70 under BGE-M3;
        # real hits sit ≥ 0.80. 0.72 drops junk without losing a single measured
        # eval hit (calibrated on the live graph, 2026-07). Set 0 to disable.
        self.min_relevance = min_relevance
        # Bottom of the lexical rescue band. `None` restores the old single-threshold floor —
        # useful for A/B-ing the change (scripts/run_eval.py --no-rescue).
        self.rescue_floor = rescue_floor

    @property
    def candidate_multiplier(self) -> int:
        return self._candidate_multiplier

    @candidate_multiplier.setter
    def candidate_multiplier(self, value: int) -> None:
        """Guarded on the SETTER, not in ``__init__``.

        The A/B harness assigns this after construction (``scripts.run_eval
        --candidate-multiplier``), so an ``__init__``-only check silently misses the exact path
        most likely to widen the pool — which is how this warning failed to fire the first time it
        was written. A guard that depends on which code path set the value is not a guard.

        Deliberately a warning, not an error: going wider is legitimate for measurement, and that
        is how the cliff was found. But the reason has to travel with the parameter, because the
        failure it causes is a scope leak in a *different* project — nothing anyone would notice
        while tuning this one.
        """
        self._candidate_multiplier = value
        if value > MEASURED_SAFE_MULTIPLIER:
            logger.warning(
                "candidate_multiplier=%d exceeds the measured-safe %d: wider pools have been "
                "measured to admit cross-project leakage (eval violations). Run "
                "scripts.run_eval and confirm violations==0 before keeping this.",
                value, MEASURED_SAFE_MULTIPLIER,
            )

    def _relevance_baseline(self) -> float:
        """Absolute anchor for rescaled relevance: the lowest similarity this engine admits."""
        if self.min_relevance <= 0:
            return COSINE_BASELINE
        if self.rescue_floor is not None:
            return min(self.rescue_floor, self.min_relevance)
        return self.min_relevance

    async def search(
        self,
        query: str,
        *,
        group_ids: list[str] | None = None,
        limit: int = 10,
        as_of: datetime | None = None,
        center_node_uuid: str | None = None,
        feedback: bool = False,
        _tool: str = "search",
        _project_id: str | None = None,
    ) -> list[Recalled]:
        """Ranked multi-strategy search over explicit scopes (``None`` = all scopes).

        ``recall`` is this with scopes composed from project/agent; the MCP
        ``search`` tool calls this directly to search across all knowledge.

        ``_tool`` / ``_project_id`` carry attribution for the query log. Underscored because they
        are telemetry, not retrieval inputs — every path funnels through here, so logging once at
        this choke point is what keeps a future caller from silently going unrecorded, but the
        cost is that recall() has to say which tool it really was.
        """
        started = time.perf_counter()
        # Strategy 1+2: semantic + graph (Graphiti hybrid; center_node_uuid adds BFS).
        multiplier = self.candidate_multiplier
        fetch = limit * multiplier
        raw = await self.searcher.search(query, group_ids, fetch, center_node_uuid)
        # Strategy 3: temporal filter (point-in-time). Archived facts are already
        # dropped by the searcher, so surviving < limit means valid hits may sit
        # past the fetch cap.
        candidates = temporal_filter(raw, as_of)

        # Back-fill (one extra round max): if filtering left us short AND the raw
        # fetch was saturated (more may exist), widen the fetch and refilter.
        if len(candidates) < limit and len(raw) >= fetch:
            fetch = limit * multiplier * 2
            logger.info(
                "recall back-fill: %d/%d valid after filter, re-fetching %d candidates",
                len(candidates), limit, fetch,
            )
            raw = await self.searcher.search(query, group_ids, fetch, center_node_uuid)
            candidates = temporal_filter(raw, as_of)

        # Similarity floor: drop confident-junk candidates whose absolute cosine is
        # below the floor. Off-topic queries return few/no results instead of a full
        # plausible-looking set. Applied after temporal filtering so we never spend
        # the floor budget on already-superseded facts.
        before_floor = len(candidates)
        candidates = apply_similarity_floor(
            candidates, self.min_relevance, query=query, rescue_floor=self.rescue_floor,
        )
        if len(candidates) < before_floor:
            logger.info(
                "similarity floor (%.2f) dropped %d/%d low-relevance candidate(s)",
                self.min_relevance, before_floor - len(candidates), before_floor,
            )

        # Node-derived signals: degree (connectivity bonus) + categorical confidence.
        # Independent queries → run concurrently.
        connectivity, node_conf = await asyncio.gather(
            self._connectivity(candidates),
            self._node_confidence(candidates),
        )

        # Rank fusion (roadmap item 17). Computed only when it carries weight — the lenses are
        # cheap but not free, and an inert mechanism should cost nothing.
        fusion = (
            rrf_fuse(lens_ranks(candidates, query)) if self.weights.fusion > 0 else None
        )

        ranked = score_facts(
            candidates, self.weights, connectivity, now=as_of,
            node_confidence=node_conf,
            fusion=fusion,
            # Relevance is rescaled from the LOWEST score we were willing to admit, so the
            # reported number stays absolute and comparable across queries (research §1.4).
            # With a rescue band active that is `rescue_floor`, not `min_relevance` — otherwise
            # every rescued fact clamps to relevance 0.0 and the rescue buys nothing.
            relevance_baseline=self._relevance_baseline(),
        )
        # Diversity pass BEFORE truncation, so near-duplicate facts don't consume the result
        # slots (research §1.3). Operates on the full candidate set, not the top-`limit`.
        diversified = mmr_rerank(
            ranked, limit, lambda_=self.mmr_lambda, max_per_source=self.max_per_source,
        )
        # MMR decides WHICH facts survive; it must not decide the order they are PRESENTED in.
        # Its selection order is by marginal value, which surfaces as a non-monotonic score
        # sequence (0.35, 0.13, 0.23, 0.30) that reads like a bug to any consumer. Re-sort by
        # composite so the contract "results are ordered best-first" still holds.
        diversified.sort(key=lambda t: t[1], reverse=True)
        hits = [
            Recalled(
                fact=f.fact, score=round(s, 4), scope=f.group_id, uuid=f.uuid,
                valid_at=f.valid_at, components={k: round(v, 4) for k, v in c.items()},
            )
            for f, s, c in diversified
        ]
        # Impressions are OPT-IN (roadmap item 14). If every read counted, the eval harness would
        # inflate the counters of the very facts it measures and UI browsing would look like agent
        # usage — the signal would be self-referential. Only a real consumption asks for this.
        if feedback and hits:
            await self._record_recall([h.uuid for h in hits])
        # Telemetry, after the work and outside its failure path: a query the author never thought
        # to write down is exactly the kind the held-out eval set needs, and there is no way to
        # recover one that was not recorded at the time.
        if self.log_queries:
            await query_log.record(
                self.redis, tool=_tool, query=query, scopes=group_ids, results=hits,
                latency_ms=(time.perf_counter() - started) * 1000.0, project_id=_project_id,
            )
        return hits

    async def recall(
        self,
        query: str,
        *,
        project_id: str | None = None,
        agent_role: str | None = None,
        limit: int = 10,
        as_of: datetime | None = None,
        center_node_uuid: str | None = None,
        feedback: bool = False,
    ) -> list[Recalled]:
        # global + cluster + project (+ agent), composed and ranked together (R5).
        # The cluster tier is what lets domain-general knowledge reach its sibling projects
        # without polluting unrelated ones (research §0).
        cluster = self._cluster_for(project_id)
        return await self.search(
            query,
            group_ids=Scope.compose(project_id, agent_role, cluster=cluster),
            limit=limit,
            as_of=as_of,
            center_node_uuid=center_node_uuid,
            feedback=feedback,
            _tool="recall",
            _project_id=project_id,
        )

    async def brief(self, project_id: str, *, use_cache: bool = True) -> Brief:
        key = f"brief:{project_id}"
        if use_cache and self.redis is not None:
            cached = await self.redis.get(key)
            if cached:
                data = json.loads(cached)
                data["cached"] = True
                return Brief(**data)

        # The cluster tier belongs in the brief too. It was wired into recall() in Wave 1 but not
        # here, so domain knowledge was invisible to the killer feature — the one place it matters
        # most, since the brief is what every session starts from.
        cluster = self._cluster_for(project_id)
        scopes = Scope.compose(project_id, cluster=cluster)   # global + cluster + project
        # "Cross-project" now means the genuinely shared tiers: global AND this project's domain.
        shared_scopes = [Scope.GLOBAL] + ([Scope.cluster(cluster)] if cluster else [])

        # Independent queries → run concurrently.
        conventions, decisions, lessons, cross, facts, runbooks = await asyncio.gather(
            self.queries.nodes_by_label([_CONVENTION], scopes, 7),
            self.queries.nodes_by_label([_DECISION], scopes, 7),
            self.queries.nodes_by_label([_LESSON], scopes, 7),
            self.queries.nodes_by_label([_PATTERN, _DECISION, _CONVENTION], shared_scopes, 7),
            # Over-fetch generously: within one project most candidates share domain vocabulary
            # and collide, so a 5x pool left the section half-empty on a real trading brief.
            self._recent_facts(scopes, _BRIEF_FACTS * _BRIEF_FACT_OVERFETCH),
            self._runbooks(scopes, _BRIEF_RUNBOOKS),
        )

        lines = {
            "active_conventions": [self._line(n) for n in conventions],
            "key_decisions": [self._line(n) for n in decisions],
            "relevant_lessons": [self._line(n) for n in self._by_severity(lessons)],
            "cross_project_knowledge": [self._line(n) for n in cross],
        }
        already = [line for group in lines.values() for line in group]

        brief = Brief(
            project_id=project_id,
            project_summary=self._summarize(project_id, conventions, decisions, lessons),
            # Spelled out rather than **lines: the splat let a typo in a key reach Brief as an
            # unexpected kwarg at runtime, and hid the four required fields from the reader.
            active_conventions=lines["active_conventions"],
            key_decisions=lines["key_decisions"],
            relevant_lessons=lines["relevant_lessons"],
            cross_project_knowledge=lines["cross_project_knowledge"],
            # Knowledge the label sections structurally cannot reach: 756 of 815 untyped entities
            # carry a real fact in their summary, and brief() queries by LABEL. Filtered against
            # the lines already present so the brief never says the same thing twice.
            recent_facts=novel_lines([f.fact for f in facts], already, _BRIEF_FACTS),
            runbooks=runbooks,
            generated_at=datetime.now(timezone.utc),
        )

        if self.redis is not None:
            await self.redis.set(key, brief.model_dump_json(), ex=self.brief_ttl)
        return brief

    async def invalidate_brief(self, project_id: str) -> None:
        if self.redis is not None:
            await self.redis.delete(f"brief:{project_id}")

    # --- helpers -------------------------------------------------------------

    async def _runbooks(self, scopes: list[str], limit: int) -> list[BriefRunbook]:
        """Procedures visible from this seat. Best-effort — a brief must render without them.

        Deliberately NOT routed through `novel_lines` like the prose sections: that filter drops a
        candidate that overlaps something already in the brief, which for a runbook would mean
        silently withholding the steps because a Convention mentioned the same topic. A procedure
        is either present in full or not present.
        """
        fetch = getattr(self.queries, "runbooks", None)
        if fetch is None:
            return []
        try:
            records = await fetch(scopes, limit)
        except Exception:  # noqa: BLE001 — never fail a brief over its newest section
            logger.warning("brief: runbook lookup failed; section omitted", exc_info=True)
            return []
        return [
            BriefRunbook(
                name=r.name, scope=r.scope, steps=r.steps, purpose=r.purpose,
                prerequisites=r.prerequisites, verified_at=r.verified_at, stale=r.is_stale(),
            )
            for r in records
        ]

    async def _recent_facts(self, scopes: list[str], limit: int) -> list[Fact]:
        """Newest valid facts in scope. Best-effort — a brief must still render without them."""
        fetch = getattr(self.queries, "recent_facts", None)
        if fetch is None:
            return []
        try:
            return await fetch(scopes, limit)
        except Exception:  # noqa: BLE001 — the brief is the killer feature; never fail it for this
            logger.warning("recent-facts lookup failed; brief omits the facts section",
                           exc_info=True)
            return []

    async def _record_recall(self, fact_uuids: list[str]) -> None:
        """Count impressions. Best-effort — a feedback write must never fail a read."""
        record = getattr(self.queries, "record_recall", None)
        if record is None:
            return
        try:
            await record(fact_uuids)
        except Exception:  # noqa: BLE001 — retrieval is the core value; feedback is bookkeeping
            logger.warning("could not record recall impressions", exc_info=True)

    def _cluster_for(self, project_id: str | None) -> str | None:
        """Resolve a project's domain cluster; never raises (a bad registry must not break recall)."""
        if not project_id or self.cluster_resolver is None:
            return None
        try:
            return self.cluster_resolver(project_id)
        except Exception:  # noqa: BLE001 — an unreadable registry degrades to global + project
            logger.warning("cluster lookup failed for %s; composing without the cluster tier",
                           project_id, exc_info=True)
            return None

    async def _node_confidence(self, facts: list[Fact]) -> dict[str, float]:
        """Categorical confidence of the candidates' endpoint nodes, as floats in [0, 1].

        Best-effort: a searcher/queries implementation without ``node_confidence`` (or a driver
        error) yields ``{}`` and every fact falls back to the 0.5 neutral — the same behaviour as
        before this signal existed, never an exception.
        """
        node_uuids = list({u for f in facts for u in (f.source_uuid, f.target_uuid) if u})
        if not node_uuids:
            return {}
        fetch = getattr(self.queries, "node_confidence", None)
        if fetch is None:
            return {}
        try:
            return await fetch(node_uuids)
        except Exception:  # noqa: BLE001 — a missing confidence signal must never fail a recall
            logger.warning("node confidence lookup failed; ranking with the neutral default",
                           exc_info=True)
            return {}

    async def _connectivity(self, facts: list[Fact]) -> dict[str, float]:
        node_uuids = list({u for f in facts for u in (f.source_uuid, f.target_uuid) if u})
        if not node_uuids:
            return {}
        degrees = await self.queries.degrees(node_uuids)
        if not degrees:
            return {}
        max_deg = max(degrees.values()) or 1
        out: dict[str, float] = {}
        for f in facts:
            d = max(degrees.get(f.source_uuid or "", 0), degrees.get(f.target_uuid or "", 0))
            out[f.uuid] = d / max_deg
        return out

    @staticmethod
    def _line(n: NodeRow) -> str:
        """One brief line: the node's summary, plus its rationale / rejected alternatives.

        The extras are what make a brief actionable rather than declarative — "we chose X"
        becomes "we chose X because Y, having rejected Z". They were populated on most Decision
        and Tool nodes but selected by no read path until 2026-07-25 (research §2.1).
        """
        base = n.summary.strip() if n.summary else n.name
        why = _trim(n.attributes.get("rationale"))
        rejected = _trim(n.attributes.get("alternatives_considered") or n.attributes.get("chosen_over"))
        extras = []
        if why:
            extras.append(f"why: {why}")
        if rejected:
            extras.append(f"rejected: {rejected}")
        return f"{base} — {'; '.join(extras)}" if extras else base

    @staticmethod
    def _by_severity(lessons: list[NodeRow]) -> list[NodeRow]:
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(lessons, key=lambda n: order.get(str(n.attributes.get("severity", "")).lower(), 4))

    @staticmethod
    def _summarize(project_id: str, conv, dec, les) -> str:
        return (
            f"{project_id}: {len(dec)} key decisions, {len(conv)} active conventions, "
            f"{len(les)} lessons on record."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Real collaborators (wrap Graphiti / Neo4j)
# ──────────────────────────────────────────────────────────────────────────────


class GraphitiSearcher:
    def __init__(self, graphiti) -> None:
        self._graphiti = graphiti

    async def search(self, query, scopes, limit, center_node_uuid=None) -> list[Fact]:
        edges = await self._graphiti.search(
            query, center_node_uuid=center_node_uuid, group_ids=scopes, num_results=limit
        )
        uuids = [e.uuid for e in edges]
        archived = await self._archived_uuids(uuids)
        # Graphiti's high-level search() returns edges ordered by its cross-encoder
        # reranker but attaches NO absolute similarity score (the reranker emits only
        # relative rank positions). An absolute cosine score is what a similarity floor
        # needs — BGE-M3's baseline self-similarity is ~0.65 even for unrelated text,
        # so only a real cosine value separates an on-topic hit (~0.85) from off-topic
        # junk (~0.68). We compute it directly: embed the query once, then batch a
        # single vector.similarity.cosine over the returned edges' fact_embedding.
        cosine, embeddings = await self._cosine_scores(query, uuids)
        return [
            Fact(
                uuid=e.uuid, fact=e.fact, group_id=e.group_id, created_at=e.created_at,
                valid_at=e.valid_at, invalid_at=e.invalid_at,
                source_uuid=e.source_node_uuid, target_uuid=e.target_node_uuid,
                confidence=(e.attributes or {}).get("confidence") if e.attributes else None,
                score=cosine.get(e.uuid, self._edge_score(e)),
                embedding=embeddings.get(e.uuid),
            )
            for e in edges
            if e.uuid not in archived  # curation-archived facts are hidden from retrieval
        ]

    async def _cosine_scores(
        self, query: str, uuids: list[str]
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Absolute cosine similarity of the query vs each returned edge's fact, plus the vectors.

        Returns ``({uuid: cosine}, {uuid: fact_embedding})`` for edges whose ``fact_embedding``
        is present. The embeddings ride along on this existing round-trip (rather than costing a
        second query) because MMR diversity needs fact-to-fact similarity, not just
        fact-to-query — see :func:`mmr_rerank`.

        Best-effort: any failure (embedder hiccup, missing vector op) yields empty maps so the
        caller falls back to ``_edge_score`` / positional relevance and MMR degrades to plain
        relevance order — the floor simply doesn't engage rather than dropping everything.
        """
        if not uuids:
            return {}, {}
        try:
            embedded = await self._graphiti.embedder.create(input_data=[query])
        except Exception:  # noqa: BLE001 — best-effort; degrade to no cosine score
            logger.warning("cosine scoring: query embed failed; ranking without a floor", exc_info=True)
            return {}, {}
        # embedder.create may return a flat vector or a list-of-vectors depending on version.
        qv = embedded[0] if embedded and isinstance(embedded[0], (list, tuple)) else embedded
        try:
            res = await self._graphiti.driver.execute_query(
                "MATCH ()-[r:RELATES_TO]->() "
                "WHERE r.uuid IN $uuids AND r.fact_embedding IS NOT NULL "
                "RETURN r.uuid AS uuid, "
                "vector.similarity.cosine(r.fact_embedding, $qv) AS sim, "
                "r.fact_embedding AS emb",
                uuids=uuids, qv=list(qv),
            )
        except Exception:  # noqa: BLE001 — vector op unavailable / driver error
            logger.warning("cosine scoring: similarity query failed; ranking without a floor", exc_info=True)
            return {}, {}
        scores = {r["uuid"]: float(r["sim"]) for r in res.records if r["sim"] is not None}
        vectors = {r["uuid"]: list(r["emb"]) for r in res.records if r["emb"] is not None}
        return scores, vectors

    @staticmethod
    def _edge_score(edge) -> float | None:
        """Best-effort similarity score for a Graphiti search edge.

        Graphiti's reranker attaches an ephemeral score on some versions/configs.
        We read it defensively: a top-level ``score`` attribute first, then an
        ``attributes['score']`` fallback. ``None`` when unavailable → the ranker
        falls back to positional relevance for that result set.
        """
        score = getattr(edge, "score", None)
        if score is None and getattr(edge, "attributes", None):
            score = edge.attributes.get("score")
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    async def _archived_uuids(self, uuids: list[str]) -> set[str]:
        if not uuids:
            return set()
        res = await self._graphiti.driver.execute_query(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.uuid IN $uuids AND r.archived = true "
            "RETURN r.uuid AS uuid",
            uuids=uuids,
        )
        return {r["uuid"] for r in res.records}


def _to_datetime(value):
    """neo4j.time.DateTime -> datetime, passing through None and already-native values."""
    return value.to_native() if hasattr(value, "to_native") else value


class Neo4jGraphQueries:
    def __init__(self, graphiti) -> None:
        self._driver = graphiti.driver

    async def nodes_by_label(self, labels, scopes, limit) -> list[NodeRow]:
        # Only valid, non-archived nodes. Return explicit scalar props — never
        # properties(n), which drags the 1024-dim name_embedding over the wire
        # just to pop it. `severity` orders brief lessons; rationale /
        # alternatives_considered / chosen_over carry the "why" and the
        # already-rejected options (research §2.1) — the most useful text in the
        # graph, and unselected by any read path before 2026-07-25.
        result = await self._driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE any(l IN labels(n) WHERE l IN $labels)
                  AND n.group_id IN $scopes
                  AND n.invalid_at IS NULL
                  AND coalesce(n.archived, false) = false
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
                   labels(n) AS labels, n.group_id AS group_id,
                   n.created_at AS created_at, n.severity AS severity,
                   n.rationale AS rationale,
                   n.alternatives_considered AS alternatives_considered,
                   n.chosen_over AS chosen_over
            ORDER BY n.created_at DESC LIMIT $limit
            """,
            labels=labels, scopes=scopes, limit=limit,
        )
        rows = []
        for r in result.records:
            created = r["created_at"]
            if hasattr(created, "to_native"):  # neo4j.time.DateTime -> datetime
                created = created.to_native()
            attributes = {"severity": r["severity"]} if r["severity"] is not None else {}
            for key in _EXTRA_NODE_ATTRS:
                if r[key] is not None:
                    attributes[key] = r[key]
            rows.append(
                NodeRow(
                    uuid=r["uuid"], name=r["name"], summary=r["summary"],
                    labels=[l for l in r["labels"] if l != "Entity"],
                    group_id=r["group_id"], created_at=created, attributes=attributes,
                )
            )
        return rows

    async def degrees(self, node_uuids) -> dict[str, int]:
        result = await self._driver.execute_query(
            """
            MATCH (n:Entity) WHERE n.uuid IN $uuids
            OPTIONAL MATCH (n)-[r:RELATES_TO]-()
            RETURN n.uuid AS uuid, count(r) AS degree
            """,
            uuids=node_uuids,
        )
        return {r["uuid"]: int(r["degree"]) for r in result.records}

    async def recent_facts(self, scopes, limit) -> list[Fact]:
        """Most recently valid, non-archived fact edges in scope.

        Ordered by ``valid_at`` (falling back to ``created_at``) so a brief leads with what was
        learned most recently. Returns only the scalar fields — never ``fact_embedding``, which
        would drag 1024 floats per row over the wire for no reason.
        """
        result = await self._driver.execute_query(
            """
            MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
            WHERE e.group_id IN $scopes
                  AND e.fact IS NOT NULL
                  AND e.invalid_at IS NULL
                  AND coalesce(e.archived, false) = false
            RETURN e.uuid AS uuid, e.fact AS fact, e.group_id AS group_id,
                   e.valid_at AS valid_at, e.created_at AS created_at
            ORDER BY coalesce(e.valid_at, e.created_at) DESC LIMIT $limit
            """,
            scopes=scopes, limit=limit,
        )
        facts = []
        for r in result.records:
            facts.append(Fact(
                uuid=r["uuid"], fact=r["fact"], group_id=r["group_id"],
                valid_at=_to_datetime(r["valid_at"]), created_at=_to_datetime(r["created_at"]),
            ))
        return facts

    async def record_recall(self, fact_uuids) -> None:
        """Increment the impression counter on each served fact (roadmap item 14).

        One UNWIND for the whole result set, so a recall costs one extra small write regardless of
        how many facts it returned.
        """
        if not fact_uuids:
            return
        await self._driver.execute_query(
            """
            UNWIND $uuids AS uuid
            MATCH ()-[e:RELATES_TO {uuid: uuid}]->()
            SET e.recalled_n = coalesce(e.recalled_n, 0) + 1,
                e.last_recalled_at = datetime()
            """,
            uuids=list(fact_uuids),
        )

    async def runbooks(self, scopes, limit: int):
        """Procedures in scope, steps intact (roadmap item 18).

        Delegates to :class:`~synapse.core.runbooks.RunbookStore` rather than inlining the Cypher,
        so the brief and the MCP/API surfaces read runbooks through exactly one query. A procedure
        that renders differently depending on which door you came in is worse than no procedure.
        """
        from synapse.core.runbooks import RunbookStore

        return await RunbookStore(self._graph_like()).list_for_scopes(list(scopes), limit=limit)

    def _graph_like(self):
        """A minimal `.driver`-shaped object for collaborators that expect one."""
        from synapse.db.neo4j_client import DirectGraph

        return DirectGraph(self._driver)

    async def node_confidence(self, node_uuids) -> dict[str, float]:
        """Endpoint-node confidence as floats, mapped from the schema's categorical Literal.

        Unknown/absent values are simply omitted, so the ranker falls back to its neutral 0.5
        for facts whose endpoints carry no confidence.
        """
        result = await self._driver.execute_query(
            """
            MATCH (n:Entity) WHERE n.uuid IN $uuids AND n.confidence IS NOT NULL
            RETURN n.uuid AS uuid, n.confidence AS confidence
            """,
            uuids=node_uuids,
        )
        out: dict[str, float] = {}
        for r in result.records:
            value = CONFIDENCE_WORDS.get(str(r["confidence"]).strip().lower())
            if value is not None:
                out[r["uuid"]] = value
        return out


def build_retrieval_engine(graphiti, *, redis=None) -> RetrievalEngine:
    """Wire the real retrieval engine. Pass a redis.asyncio client for brief caching."""
    if redis is None and settings.redis_url:
        import redis.asyncio as aioredis

        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    from synapse.core.registry import cluster_of  # lazy: keeps the module import-light

    return RetrievalEngine(
        searcher=GraphitiSearcher(graphiti),
        queries=Neo4jGraphQueries(graphiti),
        redis=redis,
        cluster_resolver=cluster_of,
    )
