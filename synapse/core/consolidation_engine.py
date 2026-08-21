"""Consolidation engine (research Wave 2) — the brain's night shift.

Synapse had **no autonomous cognition**. Curation was entirely human-triggered through a panel,
and the corpus proved it had never been used: measured 2026-07-25, ``archived`` did not exist as a
property on any of 3,011 edges, while **77 duplicate clusters** at >=0.97 similarity sat waiting and
the API's ``promotion_candidates`` computed answers that nothing could act on.

This module is the missing background process — the "sleep-time compute" / consolidation pass the
field converged on in 2026. It runs nightly on the local stack (no cloud LLM, so $0) and is
**propose-only**: it never mutates the graph. It writes ``CurationProposal`` nodes that a human (or
a later, more autonomous stage) reviews and applies. Every application routes through machinery that
is already backup-first and zero-loss-verified (R8).

Two proposal kinds, both derived from deterministic analysis:

**merge** — within-scope near-duplicate facts (>=``curation_dedup_threshold``). This is what makes
recall waste result slots on restatements; MMR hides the symptom at read time, consolidation removes
the cause. Applying one delegates to :meth:`CurationEngine.merge_duplicate`, which snapshots a
one-hop neighbourhood, supersedes temporally (never deletes, R4) and runs ``verify_no_loss``.

**promote** — a knowledge entity whose name recurs across >=2 project scopes. The target scope is
computed from the projects' clusters: one cluster => ``cluster_X``, more than one => ``global``.
This is what finally populates the cluster tier added in Wave 1.

Why promotion is entity-level and not fact-level
------------------------------------------------
The obvious design — find the same *fact* recorded in two projects and widen its scope — was
measured against the live graph and does not work. Cross-scope fact pairs number **18 at >=0.85 and
zero at >=0.90** (checked at k=40 and k=100, so neighbour-crowding is not the cause). Facts are
phrased with project-specific entities, so identical knowledge never yields near-identical fact
text: "Acme-API uses BigDecimal for money" and the Acme-Sim equivalent simply are not near-duplicates.
Recurring *entity names* are the real signal — TimescaleDB appears in four trading projects.

That also means promotion is **synthesis, not relocation**: the promotable knowledge is a new
statement ("TimescaleDB is the standard time-series store across the trading projects"), not an
existing fact moved sideways. So applying a promotion requires a ``statement``, supplied by the
reviewer, and it is stored through the normal protected write path — triage, credential redaction,
dedup and all. The engine refuses to invent the sentence itself.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from synapse.core.schema import GLOBAL_SCOPE, Scope

logger = logging.getLogger("synapse.consolidation")

# Entity labels that represent durable knowledge (not the Project bookkeeping nodes). A name
# recurring across projects only means something for these.
_KNOWLEDGE_LABELS = ["Decision", "Convention", "Lesson", "Research", "Pattern", "Tool"]

PROPOSAL_LABEL = "CurationProposal"


class ProposalKind(str, Enum):
    MERGE = "merge"
    PROMOTE = "promote"
    CONTRADICTION = "contradiction"


class ProposalStatus(str, Enum):
    OPEN = "open"
    APPLIED = "applied"
    DISMISSED = "dismissed"


class Proposal(BaseModel):
    uuid: str
    kind: ProposalKind
    status: ProposalStatus
    key: str                       # idempotency key — one proposal per real-world suggestion
    rationale: str = ""
    created_at: datetime | None = None
    applied_at: datetime | None = None
    # merge
    canonical_uuid: str | None = None
    duplicate_uuid: str | None = None
    similarity: float | None = None
    canonical_fact: str = ""
    duplicate_fact: str = ""
    differs_by: list[str] = Field(default_factory=list)   # tokens in one fact but not the other
    # promote
    name: str | None = None
    knowledge_type: str | None = None
    target_scope: str | None = None
    from_scopes: list[str] = Field(default_factory=list)


class ConsolidationRun(BaseModel):
    merges_proposed: int = 0
    promotions_proposed: int = 0
    already_known: int = 0          # suggestions that already had a proposal (any status)
    # High-cosine pairs withheld because they name different things and so are distinct
    # knowledge, not restatements (see states_different_things). Reported, never silent.
    unsafe_merges_withheld: int = 0
    contradictions_proposed: int = 0
    # Of the lexically-safe candidates, those a semantic judge rejected as not-duplicates.
    adjudicator_rejected: int = 0
    generated_at: datetime | None = None


class ProposalResult(BaseModel):
    ok: bool
    uuid: str
    action: str
    detail: str = ""
    needs_statement: bool = False   # promote/apply without a statement


def _native(value):
    return value.to_native() if hasattr(value, "to_native") else value


# --- merge safety: cosine cannot tell a restatement from a different VALUE --------------------
# Found the first time this engine ran against the live graph (2026-07-25). At >=0.97 cosine it
# proposed merging pairs that are NOT duplicates but distinct facts sharing a sentence pattern:
#
#   0.9823  "vectorbt compiles hot paths through llvmlite 0.47.0"
#        vs "vectorbt compiles hot paths through numba 0.65.1"        <- different library+version
#   0.9834  "...had ITB as one of 9 pinned sector ETFs..."
#        vs "...had IBB as one of 9 pinned sector ETFs..."            <- different ETF
#   0.9824  "Soniox is confirmed for Hebrew STT at $0.12/hr"
#        vs "Soniox is the only vendor offering Hebrew<->English code-switching"
#
# Short factual sentences differing in one entity or number are ~0.98 similar, so raising the
# threshold cannot fix this — true restatements sit in the same range. Applying those merges
# would DESTROY knowledge, which R8 forbids outright.
#
# So a merge candidate whose differing tokens carry a *value* (a number, version, price, ticker or
# acronym) is not proposed at all. This errs toward missing a merge, which costs one recall slot,
# over making a wrong one, which loses a fact permanently. Genuine restatements differ only in
# phrasing, and those still get through.
_TOKEN = re.compile(r"[A-Za-z0-9$€£][A-Za-z0-9$€£.,:/%_+-]*")
_HAS_DIGIT = re.compile(r"\d")
# 2-6 character all-caps runs: tickers (ITB/IBB/SPY), vendor/protocol acronyms, env names.
_ACRONYM = re.compile(r"^[A-Z][A-Z0-9]{1,5}$")


def _tokens(fact: str) -> list[str]:
    return _TOKEN.findall(fact or "")


def discriminating_tokens(fact_a: str, fact_b: str) -> list[str]:
    """Tokens present in one fact but not the other — what a reviewer needs to see."""
    a, b = set(_tokens(fact_a)), set(_tokens(fact_b))
    return sorted(a.symmetric_difference(b))


def carries_distinct_values(fact_a: str, fact_b: str) -> bool:
    """True when the two facts differ by a VALUE (number, version, price, ticker, acronym).

    NOTE the scope of this test: it was built for pairs a similarity search already called
    near-identical, where only a handful of tokens differ. Between two UNRELATED facts nearly
    every token differs, so something digit-bearing almost always appears and this returns True.
    Do not use it alone as evidence of a contradiction — pair it with ``subject_overlap``. See
    ``invalidation_is_credible``.
    """
    for token in discriminating_tokens(fact_a, fact_b):
        stripped = token.strip(".,:/%_+-")
        if not stripped:
            continue
        if _HAS_DIGIT.search(stripped) or _ACRONYM.match(stripped):
            return True
    return False


# Grammatical scaffolding. A difference confined to these words (or to morphological variants of
# shared words) is a rephrasing; a difference in any remaining word names a different thing.
_FUNCTION_WORDS = frozenset(
    "a an the this that these those it its his her their our your my and or but nor so yet for "
    "to of in on at by with from as into onto over under within without across per via about "
    "is are was were be been being am do does did done has have had having will would shall "
    "should can could may might must not no nor if then than when while because since although "
    "though whether both each any all some only just also still even other another same such "
    "which who whom whose what where how there here now already always never ever one".split()
)
_SUFFIXES = ("'s", "ies", "es", "s", "ed", "ing", "ly")


def _stem(token: str) -> str:
    """Crude morphological normalizer so flag/flags and affect/affecting cancel out."""
    t = token.lower().strip(".,:;/%_+-'\"()[]")
    for suffix in _SUFFIXES:
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            return t[: -len(suffix)]
    return t


def _content_counts(fact: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for token in _tokens(fact):
        stem = _stem(token)
        if stem and stem not in _FUNCTION_WORDS:
            counts[stem] += 1
    return counts


def content_difference(fact_a: str, fact_b: str) -> tuple[list[str], list[str]]:
    """Content words present in only one fact, after cancelling grammatical variants.

    Returns ``(only_in_a, only_in_b)``. Empty on both sides means the two facts are built from
    the same content words *with the same multiplicities*, so they differ only in phrasing or
    word order.

    Counting matters, not just set membership. This pair has identical word sets:

        "The LightGBM + XGBoost ensemble includes XGBoost as a component."
        "The LightGBM + XGBoost ensemble includes LightGBM as a component."

    but different counts (XGBoost x2/x1, LightGBM x1/x2), and they state different things.
    """
    counts_a, counts_b = _content_counts(fact_a), _content_counts(fact_b)
    only_a = sorted((counts_a - counts_b).elements())
    only_b = sorted((counts_b - counts_a).elements())
    return only_a, only_b


def states_different_things(fact_a: str, fact_b: str) -> bool:
    """The merge safety gate: True when the pair is distinct knowledge, not a restatement.

    Strengthened after the SECOND live run (2026-07-25). Gating only on numbers and acronyms still
    let through pairs that name genuinely different things in an identical sentence frame:

      0.9707  "...contains a technical analysis node as one of five parallel..."
           vs "...contains a fundamental analysis node as one of five parallel..."
      0.9709  "The Unmanaged-position guard's soft debounce is based on ticks."
           vs "The Unmanaged-position guard's hard debounce is based on ticks."
      0.9712  "'flyway_schema_history_watchtower' is an example of..."
           vs "'flyway_schema_history_gateway' is an example of..."
      0.9702  "Sonnet implemented the unified React Flow canvas in Acme-Flow Sprint 1."
           vs "Sonnet implemented the unified React Flow canvas AND backend reliability tasks..."

    Antonym pairs, sibling categories, distinct identifiers and supersets all read as ~0.97 cosine
    because the sentence frame dominates the embedding. The only difference cosine is blind to and
    a reviewer is not: **which content words each fact actually contains.**

    So a merge is proposed only when the two facts share their entire content vocabulary, counts
    included. That is deliberately strict — it will decline genuine restatements that introduce a
    synonym — because a declined merge costs one recall slot while a wrong merge loses a fact
    permanently (R8).

    **This gate is necessary but not sufficient**, and it cuts both ways. A pair can share its
    entire content multiset and still need judgement, because the difference is in grammatical
    ROLE rather than vocabulary:

        "Walk-forward validation is performed using backtrader (alongside vectorbt)"
     vs "Walk-forward validation is performed using vectorbt (alongside backtrader)"
        "MES feed went silent simultaneously with NQ, ES, and MNQ at 13:00 ET"
     vs "NQ feed went silent simultaneously with ES, MNQ, and MES at 13:00 ET"

    Both of those look like role swaps but are in fact genuine duplicates, because "alongside" and
    "simultaneously with" are *symmetric* relations — which of the four feeds is the grammatical
    subject carries no information. The semantic judge confirmed both as duplicates on the live
    graph, and it is right to. A bag-of-words method cannot reach that conclusion either way: it
    sees only that the vocabulary matches.

    So the lexical gate is the cheap pre-filter — it removed ~79% of candidates for free — and the
    optional ``adjudicator`` (the write pipeline's duplicate/contradiction/distinct judge) settles
    the permutations the vocabulary alone cannot decide.
    """
    only_a, only_b = content_difference(fact_a, fact_b)
    return bool(only_a or only_b)


# --- credibility of an automatic invalidation -----------------------------------------------------
#
# Graphiti's ``add_episode`` invalidates existing edges it judges contradicted by the new episode.
# That judgement is made by an LLM over candidates a similarity search supplied, and it OVER-REACHES.
# Measured on the live graph 2026-07-27: writing ONE convention about a service's port numbers
# invalidated five valid edges — three that the new fact merely restated more completely, and two
# about TimescaleDB, which shares no subject with gateways or ports at all. Nothing errored; the
# write reported success. An earlier instance sat undetected for seven weeks: a cluster's move to a
# containerized gateway invalidated the UPSTREAM VENDOR DEFAULT ports alongside the local convention
# that had genuinely changed. Vendor defaults are not refuted by us deploying differently, and that
# mis-invalidation is what made the corresponding eval case fail.
#
# A contradiction requires TWO things at once: the same subject, stated with a different value.
# Each half alone is worthless here, and MEASURED on three labelled groups they fail in opposite
# directions:
#
#   group                                        n    subject_overlap   carries_distinct_values
#   genuine contradictions (item-16 sweep)       2    0.917 – 0.933     2/2  yes
#   the 5 wrongly-invalidated edges              5    0.091 – 0.438     5/5  yes   <- value alone
#                                                                                     permits ALL
#   merge proposals, i.e. true restatements     20    1.000             2/20 yes   <- overlap alone
#                                                                                     permits ALL
#
# Overlap separates cleanly (0.438 vs 0.917) and the value test excludes the restatements, so the
# CONJUNCTION admits the positives and rejects both kinds of negative. The threshold sits near the
# midpoint of the measured gap rather than at either class's edge, because two positives is a thin
# sample and a boundary-hugging threshold would be fitting noise.
_MIN_SUBJECT_OVERLAP = 0.70


def subject_overlap(fact_a: str, fact_b: str) -> float:
    """Shared content words as a fraction of the LARGER fact's content words.

    Dividing by the larger side is the conservative choice: a short fact that happens to be a
    vocabulary subset of a long one is not thereby "about the same subject". Using the smaller
    side would score ``TimescaleDB is used in acme-sim`` highly against any long acme-sim fact.
    """
    counts_a, counts_b = _content_counts(fact_a), _content_counts(fact_b)
    if not counts_a or not counts_b:
        return 0.0
    shared = sum((counts_a & counts_b).values())
    return shared / max(sum(counts_a.values()), sum(counts_b.values()))


def invalidation_is_credible(old_fact: str, new_fact: str) -> bool:
    """Could *new_fact* genuinely have superseded *old_fact*? Same subject, different value.

    This is a VETO used on the write path, not a detector: returning False means "do not believe
    the automatic invalidation", and the effect is that a fact stays valid. So its failure mode is
    keeping a fact that should have been retired — recoverable, and visible in the contradiction
    review queue — rather than silently losing knowledge, which R8 forbids.

    Necessary but NOT sufficient, exactly like ``states_different_things``. Two of the twenty
    measured restatements share their whole vocabulary AND carry a value difference (the symmetric
    ``MES feed went silent simultaneously with NQ, ES, and MNQ at 13:00 ET`` permutations, which the
    semantic judge correctly called duplicates). Those pass this gate, so it would not rescue them.
    Deciding *which* of two same-subject facts is true is a world-judgement and stays with the
    contradiction review queue.
    """
    # MEASURED AND REJECTED, 2026-08-20, against 40 hand-labelled retirements: broadening
    # `carries_distinct_values` to also accept a NAMED-ENTITY substitution (Suno -> YouTube Audio
    # Library, Kafka -> Spring Modulith — the shape most supersessions in a software corpus
    # actually take) looks obviously right and is strictly worse. It introduced 3 silent losses of
    # true facts and recovered 0 stale ones, taking total error from 11/40 to 14/40. R8's asymmetry
    # is what makes that a clear loss rather than a wash. Do not re-add it without new labels.
    return (
        subject_overlap(old_fact, new_fact) >= _MIN_SUBJECT_OVERLAP
        and carries_distinct_values(old_fact, new_fact)
    )


def _fold_relation(name: str | None) -> str:
    """Case- and separator-folded relation name: RUNS_ON, runs-on and runsOn are one name."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def could_replace(
    new_edge: tuple[str, str, str | None], retired: tuple[str, str, str | None]
) -> bool:
    """Could *new_edge* STRUCTURALLY have replaced *retired*? Each is (source, target, relation).

    The lexical vetoes above ask whether two sentences are about the same thing. This asks the
    prior question the graph can answer without reading anything: does the new edge even describe
    the same relationship? Two shapes qualify, and only two:

    * the same two entities, in either direction — extraction direction is not stable, so a
      restatement can legitimately arrive reversed;
    * one endpoint held in the SAME position plus the same relation name, which is what a rename
      or a re-point looks like.

    Anything else is a statement about a different relationship, and a fact about a different
    relationship cannot supersede this one however similar the prose happens to read.

    This mirrors the guard proposed upstream in getzep/graphiti#1729. We apply it as a veto after
    the fact rather than as a pre-filter, because we do not control Graphiti's candidate search —
    but the test is the same, which is deliberate: if that PR lands, this becomes a redundant
    second line rather than a divergent one.
    """
    new_src, new_dst, new_name = new_edge
    old_src, old_dst, old_name = retired
    if {new_src, new_dst} == {old_src, old_dst}:
        return True
    shares_position = new_src == old_src or new_dst == old_dst
    return shares_position and _fold_relation(new_name) == _fold_relation(old_name)


class ConsolidationEngine:
    def __init__(
        self,
        graphiti,
        curation,
        *,
        cluster_resolver=None,
        remember=None,
        adjudicator=None,
        now: datetime | None = None,
    ) -> None:
        self._driver = graphiti.driver
        self.curation = curation
        # project_id -> cluster name. Injected (not imported) so this stays registry-agnostic
        # and unit-testable, matching RetrievalEngine.
        self.cluster_resolver = cluster_resolver
        # KnowledgeEngine.remember — used only when APPLYING a promotion, so the new statement
        # goes through triage/redaction/dedup like any other write.
        self._remember = remember
        # Optional semantic judge with `.adjudicate(new, existing) -> Adjudication`. The write
        # pipeline's ClaudeTriage satisfies this and runs on local gemma when cloud credits are
        # out, so the final gate costs nothing. Without it, only the lexical gate applies and
        # role-swap pairs can still be proposed — so the reviewer stays the last line of defence.
        self._adjudicator = adjudicator
        self._now = now

    def _utcnow(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    # --- generation (read-only analysis -> proposals) -------------------------

    async def propose(
        self, *, max_merges: int = 50, max_promotions: int = 25, max_contradictions: int = 20,
    ) -> ConsolidationRun:
        """Generate proposals from the current graph. Idempotent; never mutates knowledge."""
        run = ConsolidationRun(generated_at=self._utcnow())
        merges, known_m, withheld, rejected = await self._propose_merges(max_merges)
        promos, known_p = await self._propose_promotions(max_promotions)
        contras, known_c = await self._sweep_contradictions(max_contradictions)
        run.merges_proposed = merges
        run.promotions_proposed = promos
        run.already_known = known_m + known_p + known_c
        run.contradictions_proposed = contras
        run.unsafe_merges_withheld = withheld
        run.adjudicator_rejected = rejected
        logger.info(
            "consolidation: %d merge + %d promote + %d contradiction proposals created "
            "(%d already known, %d withheld by the lexical gate, %d rejected by the adjudicator)",
            merges, promos, contras, run.already_known, withheld, rejected,
        )
        return run

    async def _propose_merges(self, limit: int) -> tuple[int, int, int, int]:
        clusters = await self.curation.find_duplicates()
        created = known = withheld = rejected = 0
        for cluster in clusters:
            for dup in cluster.duplicates:
                if created >= limit:
                    logger.info("merge proposals capped at %d; more duplicates remain", limit)
                    return created, known, withheld, rejected
                # SAFETY GATE (R8): a value difference means these are distinct facts that merely
                # share a sentence pattern. Never propose destroying one of them.
                if states_different_things(cluster.canonical.fact, dup.fact):
                    withheld += 1
                    only_a, only_b = content_difference(cluster.canonical.fact, dup.fact)
                    logger.info(
                        "withholding merge %s/%s (cosine %.4f): distinct knowledge, not a "
                        "restatement — content differs %s vs %s",
                        cluster.canonical.uuid, dup.uuid, cluster.max_similarity,
                        only_a[:5] or "-", only_b[:5] or "-",
                    )
                    continue
                # Semantic gate for what the lexical one cannot see: role swaps and permutations.
                if not await self._adjudged_duplicate(cluster.canonical.fact, dup.fact):
                    rejected += 1
                    continue
                differs = discriminating_tokens(cluster.canonical.fact, dup.fact)
                fresh = await self._persist(
                    key=f"merge:{cluster.canonical.uuid}:{dup.uuid}",
                    kind=ProposalKind.MERGE,
                    rationale=(
                        f"cosine {cluster.max_similarity:.4f} within {cluster.scope} — "
                        f"restatement of an existing fact, so it consumes a recall slot without "
                        f"adding information. Wording differs by: "
                        f"{', '.join(differs[:8]) or '(nothing but punctuation)'}"
                    ),
                    props={
                        "canonical_uuid": cluster.canonical.uuid,
                        "duplicate_uuid": dup.uuid,
                        "similarity": float(cluster.max_similarity),
                        "canonical_fact": cluster.canonical.fact,
                        "duplicate_fact": dup.fact,
                        "differs_by": differs[:12],
                    },
                )
                created += fresh
                known += 1 - fresh
        return created, known, withheld, rejected

    async def _adjudged_duplicate(self, canonical: str, duplicate: str) -> bool:
        """Ask the semantic judge whether these really are the same knowledge.

        Returns True when there is no adjudicator (lexical gate only) or the judge says
        ``duplicate``. Any judge failure returns **False** — fail closed: an unavailable judge must
        not silently downgrade this to lexical-only and start proposing role-swap merges (R8).
        """
        if self._adjudicator is None:
            return True
        try:
            verdict = await self._adjudicator.adjudicate(duplicate, canonical)
        except Exception:  # noqa: BLE001 — a judge outage must not produce unsafe proposals
            logger.warning("adjudicator failed; withholding this merge", exc_info=True)
            return False
        relation = getattr(verdict, "relation", None)
        value = getattr(relation, "value", relation)
        if value != "duplicate":
            logger.info("adjudicator rejected a lexically-safe merge candidate as '%s'", value)
            return False
        return True

    async def _propose_promotions(self, limit: int) -> tuple[int, int]:
        res = await self._driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.group_id STARTS WITH 'project_'
              AND any(l IN labels(n) WHERE l IN $klabels)
              AND n.invalid_at IS NULL AND coalesce(n.archived, false) = false
            WITH n.name AS name,
                 head([l IN labels(n) WHERE l <> 'Entity']) AS label,
                 collect(DISTINCT n.group_id) AS scopes
            WHERE size(scopes) >= 2
            RETURN name, label, scopes
            ORDER BY size(scopes) DESC, name LIMIT $limit
            """,
            klabels=_KNOWLEDGE_LABELS, limit=limit,
        )
        created = known = 0
        for r in res.records:
            scopes = list(r["scopes"])
            target = self._target_scope([s.removeprefix("project_") for s in scopes])
            if target is None:
                continue
            projects = ", ".join(sorted(s.removeprefix("project_") for s in scopes))
            fresh = await self._persist(
                key=f"promote:{r['name']}:{target}",
                kind=ProposalKind.PROMOTE,
                rationale=(
                    f"'{r['name']}' recurs in {len(scopes)} projects ({projects}) — shared "
                    f"knowledge currently invisible from its sibling projects. Promote a "
                    f"synthesized statement to {target}."
                ),
                props={
                    "name": r["name"],
                    "knowledge_type": (r["label"] or "Entity").lower(),
                    "target_scope": target,
                    "from_scopes": scopes,
                },
            )
            created += fresh
            known += 1 - fresh
        return created, known

    @staticmethod
    def _proposal_projects(proposal: Proposal) -> list[str]:
        return [s.removeprefix("project_") for s in (proposal.from_scopes or [])]

    async def _sweep_contradictions(self, limit: int) -> tuple[int, int]:
        """Adjudicate the review band for conflicting knowledge (roadmap item 16, sweep half).

        The write path only judges a NEW write against its neighbours, so nothing ever re-examined
        knowledge already sitting side by side. The corpus shows the result: **7 Contradicts edges
        across 3,039 facts** — contradiction was effectively undetected. ``find_review_pairs``
        already surfaces the gray band [``curation_review_floor``, ``curation_dedup_threshold``);
        this asks the judge which of those pairs actually conflict.

        The lexical merge gate does NOT apply here, and that is deliberate: a pair differing by a
        value ("port 8080" vs "port 9090") is exactly what a contradiction looks like. The gate
        exists to stop destructive merges; a contradiction proposal destroys nothing, it flags.

        Without an adjudicator this is a no-op rather than a guess — flagging random similar pairs
        as contradictions would poison the review queue.
        """
        if self._adjudicator is None or limit <= 0:
            return 0, 0
        pairs = await self.curation.find_review_pairs()
        created = known = 0
        for pair in pairs:
            if created >= limit:
                logger.info("contradiction sweep capped at %d proposals", limit)
                break
            # PRE-FILTER, the mirror image of the merge gate. A contradiction is the same
            # single-valued thing given two different values, so a VALUE difference (number,
            # version, ticker, acronym) is a PREREQUISITE here — exactly the signal that
            # DISQUALIFIES a merge. Pairs differing only by a set member cannot conflict:
            # "Haiku handles the sentiment node" and "...the technical node" are both true.
            #
            # This is deterministic on purpose. The local judge got that class wrong on real data,
            # and sharpening its prompt only took the queue from 9 to 6 while leaving the same
            # false positives — so the filter, not the model, is what makes the queue reviewable.
            if not carries_distinct_values(pair.a.fact, pair.b.fact):
                continue
            try:
                verdict = await self._adjudicator.adjudicate(pair.b.fact, pair.a.fact)
            except Exception:  # noqa: BLE001 — a judge outage must not abort the whole pass
                logger.warning("contradiction sweep: adjudicator failed on a pair", exc_info=True)
                continue
            # Tolerate both an enum and a bare string verdict, exactly as _adjudged_duplicate does —
            # the real ClaudeTriage returns Relation, other clients may return the plain value.
            relation = getattr(verdict, "relation", None)
            if getattr(relation, "value", relation) != "contradiction":
                continue
            fresh = await self._persist(
                key=f"contradiction:{pair.a.uuid}:{pair.b.uuid}",
                kind=ProposalKind.CONTRADICTION,
                rationale=(
                    f"cosine {pair.similarity:.4f} in {pair.scope} and the judge calls these "
                    f"CONTRADICTORY, not restatements. One of them is probably stale — resolve with "
                    f"`update` (which supersedes with history) rather than deleting either."
                ),
                props={
                    "canonical_uuid": pair.a.uuid,
                    "duplicate_uuid": pair.b.uuid,
                    "similarity": float(pair.similarity),
                    "canonical_fact": pair.a.fact,
                    "duplicate_fact": pair.b.fact,
                },
            )
            created += fresh
            known += 1 - fresh
        return created, known

    def _target_scope(self, project_ids: list[str]) -> str | None:
        """Where shared knowledge belongs: one cluster => that cluster, otherwise global.

        Returns ``None`` when there is nothing to widen to (fewer than two projects).
        """
        if len(project_ids) < 2:
            return None
        if self.cluster_resolver is None:
            return GLOBAL_SCOPE
        clusters = set()
        for pid in project_ids:
            try:
                clusters.add(self.cluster_resolver(pid))
            except Exception:  # noqa: BLE001 — an unreadable registry must not abort the scan
                clusters.add(None)
        # A single real cluster => the domain tier. Anything else (several clusters, or a mix of
        # clustered and unclustered projects) reaches beyond one domain => global.
        if len(clusters) == 1:
            only = next(iter(clusters))
            return Scope.cluster(only) if only else GLOBAL_SCOPE
        return GLOBAL_SCOPE

    async def _persist(self, *, key: str, kind: ProposalKind, rationale: str, props: dict) -> int:
        """MERGE a proposal on its idempotency key. Returns 1 if newly created, else 0.

        ``ON MATCH`` deliberately touches nothing: a proposal the human already **dismissed** must
        stay dismissed, or every nightly run would resurrect the same rejected suggestion — the
        failure mode that makes review queues get ignored.
        """
        res = await self._driver.execute_query(
            f"""
            MERGE (p:{PROPOSAL_LABEL} {{key: $key}})
            ON CREATE SET p.uuid = randomUUID(), p.kind = $kind, p.status = $open,
                          p.rationale = $rationale, p.created_at = $now, p += $props
            RETURN p.uuid AS uuid, p.status AS status,
                   p.created_at = $now AS created
            """,
            key=key, kind=kind.value, open=ProposalStatus.OPEN.value,
            rationale=rationale, now=self._utcnow().isoformat(), props=props,
        )
        return 1 if (res.records and res.records[0]["created"]) else 0

    # --- review inbox --------------------------------------------------------

    async def list_proposals(
        self, status: str | None = ProposalStatus.OPEN.value, limit: int = 100,
        kind: str | None = None,
    ) -> list[Proposal]:
        filters = ["1=1"]
        params: dict = {"limit": limit}
        if status:
            filters.append("p.status = $status")
            params["status"] = status
        if kind:
            filters.append("p.kind = $kind")
            params["kind"] = kind
        res = await self._driver.execute_query(
            f"""
            MATCH (p:{PROPOSAL_LABEL}) WHERE {" AND ".join(filters)}
            RETURN p AS p ORDER BY p.created_at DESC LIMIT $limit
            """,
            **params,
        )
        return [self._to_proposal(dict(r["p"])) for r in res.records]

    @staticmethod
    def _to_proposal(node: dict) -> Proposal:
        for key in ("created_at", "applied_at"):
            value = _native(node.get(key))
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value)
                except ValueError:
                    value = None
            node[key] = value
        node.setdefault("from_scopes", [])
        return Proposal(**{k: v for k, v in node.items() if k in Proposal.model_fields})

    async def _get(self, uuid: str) -> Proposal | None:
        res = await self._driver.execute_query(
            f"MATCH (p:{PROPOSAL_LABEL} {{uuid: $uuid}}) RETURN p AS p", uuid=uuid,
        )
        return self._to_proposal(dict(res.records[0]["p"])) if res.records else None

    async def _set_status(self, uuid: str, status: ProposalStatus) -> None:
        await self._driver.execute_query(
            f"""
            MATCH (p:{PROPOSAL_LABEL} {{uuid: $uuid}})
            SET p.status = $status,
                p.applied_at = CASE WHEN $status = $applied THEN $now ELSE p.applied_at END
            """,
            uuid=uuid, status=status.value, applied=ProposalStatus.APPLIED.value,
            now=self._utcnow().isoformat(),
        )

    async def dismiss(self, uuid: str) -> ProposalResult:
        proposal = await self._get(uuid)
        if proposal is None:
            return ProposalResult(ok=False, uuid=uuid, action="dismiss", detail="proposal not found")
        await self._set_status(uuid, ProposalStatus.DISMISSED)
        return ProposalResult(ok=True, uuid=uuid, action="dismiss",
                              detail="dismissed; it will not be proposed again")

    # --- application (the only path that touches knowledge) ------------------

    async def apply(self, uuid: str, *, statement: str | None = None) -> ProposalResult:
        """Apply one proposal. Merges reuse the backup-first, zero-loss-verified curation path.

        Promotions need a ``statement`` — see the module docstring: promotion is synthesis, and
        this engine will not invent the sentence on the reviewer's behalf.
        """
        proposal = await self._get(uuid)
        if proposal is None:
            return ProposalResult(ok=False, uuid=uuid, action="apply", detail="proposal not found")
        if proposal.status is not ProposalStatus.OPEN:
            return ProposalResult(ok=False, uuid=uuid, action="apply",
                                  detail=f"proposal is already {proposal.status.value}")

        if proposal.kind is ProposalKind.MERGE:
            return await self._apply_merge(proposal)
        if proposal.kind is ProposalKind.CONTRADICTION:
            # Resolving a contradiction means deciding WHICH fact is now true — a judgement about
            # the world, not about the graph. Applying one automatically would either delete live
            # knowledge or silently pick a winner. The reviewer resolves it with `update` (which
            # supersedes with history, R4) and then dismisses the proposal.
            return ProposalResult(
                ok=False, uuid=proposal.uuid, action="contradiction",
                detail=(
                    "a contradiction cannot be applied automatically — decide which fact is "
                    "current and call `update` on the stale one, then dismiss this proposal. "
                    f"Facts: {proposal.canonical_uuid} vs {proposal.duplicate_uuid}."
                ),
            )
        return await self._apply_promote(proposal, statement)

    async def _apply_merge(self, proposal: Proposal) -> ProposalResult:
        result = await self.curation.merge_duplicate(
            proposal.canonical_uuid, proposal.duplicate_uuid,
        )
        if not result.ok:
            return ProposalResult(ok=False, uuid=proposal.uuid, action="merge",
                                  detail=result.detail)
        await self._set_status(proposal.uuid, ProposalStatus.APPLIED)
        return ProposalResult(ok=True, uuid=proposal.uuid, action="merge",
                              detail=f"{result.detail} (backup: {result.backup_path})")

    async def _apply_promote(self, proposal: Proposal, statement: str | None) -> ProposalResult:
        if not statement or not statement.strip():
            return ProposalResult(
                ok=False, uuid=proposal.uuid, action="promote", needs_statement=True,
                detail=(
                    f"promotion to {proposal.target_scope} needs a statement of the shared "
                    f"knowledge about '{proposal.name}'. Promotion is synthesis, not relocation. "
                    f"State the knowledge ONCE and do not list the projects — they are already "
                    f"recorded in from_scopes, and enumerating them makes the extractor emit one "
                    f"near-identical fact per project."
                ),
            )
        # Guard against the fan-out that enumeration causes (learned the hard way, 2026-07-25).
        # A first promotion whose statement listed all four trading projects was decomposed by the
        # extractor into "TimescaleDB is used in acme-api", "... in acme-sim", "... in acme-data",
        # "... in acme-etl" — four zero-information facts that then outranked the real
        # TimescaleDB hypertable gotcha and cost 12% MRR on the eval set. The projects are metadata,
        # not knowledge.
        listed = [p for p in self._proposal_projects(proposal) if p and p.lower() in statement.lower()]
        if len(listed) >= 3:
            return ProposalResult(
                ok=False, uuid=proposal.uuid, action="promote", needs_statement=True,
                detail=(
                    f"the statement enumerates {len(listed)} project names ({', '.join(listed[:4])}). "
                    f"That makes the extractor emit one near-identical fact per project, which "
                    f"crowds out substantive knowledge at recall time. State what is shared about "
                    f"'{proposal.name}' once, without listing the projects."
                ),
            )
        if self._remember is None:
            return ProposalResult(ok=False, uuid=proposal.uuid, action="promote",
                                  detail="no write path wired into the consolidation engine")
        target = proposal.target_scope or GLOBAL_SCOPE
        # NOT force=True, and dedup is widened to the tiers ABOVE the target. The first real
        # promotion used force and stored knowledge that `global` already contained ("TimescaleDB
        # (PostgreSQL hypertables) is the chosen time-series store"), because dedup only compares
        # within the target scope — and a promotion's target is typically EMPTY, which is the whole
        # point of promoting. Both gates now apply, so a redundant promotion returns DUPLICATE
        # instead of adding noise.
        kwargs: dict = {
            "source": "consolidation",
            "dedup_scopes": self._promotion_dedup_scopes(target),
        }
        if target.startswith("cluster_"):
            kwargs["cluster"] = target.removeprefix("cluster_")
        # else: global — remember() with neither project nor cluster resolves to global scope.
        write = await self._remember(statement.strip(), **kwargs)

        outcome = getattr(getattr(write, "outcome", None), "value", getattr(write, "outcome", "?"))
        scope = getattr(write, "scope", target)

        # A rejected statement is a signal to the reviewer, not a completed promotion — leave the
        # proposal OPEN so they can rewrite it.
        if outcome == "rejected":
            return ProposalResult(
                ok=False, uuid=proposal.uuid, action="promote", needs_statement=True,
                detail=(
                    f"the write filter rejected this statement: "
                    f"{getattr(write, 'reason', 'no reason given')}"
                ),
            )

        await self._set_status(proposal.uuid, ProposalStatus.APPLIED)
        if outcome == "duplicate":
            # Nothing was stored, but the promotion's intent is already satisfied at the wider
            # tier, so it is done and must not be proposed again.
            return ProposalResult(
                ok=True, uuid=proposal.uuid, action="promote",
                detail=(
                    f"already covered at or above {target} — nothing stored "
                    f"(duplicate of {getattr(write, 'duplicate_of', None)}). "
                    f"The knowledge is reachable from the sibling projects already."
                ),
            )
        return ProposalResult(
            ok=True, uuid=proposal.uuid, action="promote",
            detail=f"stored at {scope} ({outcome})",
        )

    @staticmethod
    def _promotion_dedup_scopes(target: str) -> list[str]:
        """Scopes a promotion must not duplicate: the target plus every tier ABOVE it.

        Promoting into ``cluster_trading`` is pointless if ``global`` already states the same
        thing, because retrieval composes global for every project anyway. Tiers BELOW are not
        included — a project-contextualized restatement is legitimately distinct knowledge (see
        WritePipeline._dedup_scopes).
        """
        if target == GLOBAL_SCOPE:
            return [GLOBAL_SCOPE]
        return [target, GLOBAL_SCOPE]


def build_consolidation_engine(graphiti, *, remember=None, adjudicate=True) -> ConsolidationEngine:
    """Wire the real consolidation engine.

    The adjudicator is the write pipeline's own triage judge, which routes through
    ``haiku_or_local`` — so with cloud credits exhausted it runs on local gemma and the nightly
    pass still costs nothing.
    """
    from synapse.config import settings
    from synapse.core.curation_engine import build_curation_engine
    from synapse.core.registry import cluster_of
    from synapse.core.write_pipeline import ClaudeTriage

    adjudicator = None
    if adjudicate:
        adjudicator = ClaudeTriage(settings.anthropic_api_key, settings.triage_model)

    return ConsolidationEngine(
        graphiti,
        build_curation_engine(graphiti),
        cluster_resolver=cluster_of,
        remember=remember,
        adjudicator=adjudicator,
    )
