"""Synapse knowledge model — entity types, edge types, and the scope model.

This encodes plan Part 3 as **Graphiti custom types**. Two things map onto
Graphiti's native machinery rather than being re-implemented:

- **Temporal model** (`valid_from` / `valid_until`): native to Graphiti — every
  *fact edge* carries bi-temporal `valid_at` / `invalid_at`, and supersession is
  handled automatically when a new fact contradicts an old one. We therefore do
  **not** add temporal fields to entities; we lean on the engine (R4).
- **Scope** (`global` / `project:X` / `agent:Y`): maps to Graphiti's `group_id`
  partition. Scope is the episode's `group_id`, not a node attribute — see the
  ``Scope`` helpers below. Retrieval composes scopes by passing multiple
  `group_ids` (R5).

Graphiti already provides `uuid`, `name`, `summary`, `group_id`, `created_at`
and embeddings on every node, so entity classes below define **only the extra
attributes**. Field descriptions are load-bearing: Graphiti's extractor (Claude)
reads them to decide what to pull out, so they are written as extraction prompts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# Scope model — maps to Graphiti group_id (plan Part 3 "Scope Model", R5)
# ──────────────────────────────────────────────────────────────────────────────

GLOBAL_SCOPE = "global"


class Scope:
    """Helpers for the `group_id` strings that partition knowledge by scope.

    A piece of knowledge lives in exactly one scope (the episode's `group_id`).
    Retrieval *composes* scopes by querying several group_ids at once.
    """

    GLOBAL = GLOBAL_SCOPE

    # Graphiti group_ids allow only [A-Za-z0-9_-] (no colons), so scope type and
    # id are joined with an underscore: "project_acme-store", "agent_planner".
    @staticmethod
    def project(project_id: str) -> str:
        return f"project_{project_id}"

    @staticmethod
    def agent(role: str) -> str:
        return f"agent_{role}"

    @staticmethod
    def cluster(name: str) -> str:
        """The domain tier between ``project`` and ``global`` (e.g. ``cluster_trading``).

        Added 2026-07-25 (research §0). Measuring the live graph showed the hierarchy was
        degenerate: 95.5% of nodes sat in ``project_*``, ``global`` held 4.5%, ``agent_*`` was
        empty, and only **30 of 3,011 edges (1.0%) crossed projects**. The cause was a missing
        tier — knowledge that is domain-general but not universal had nowhere to live. A lesson
        like "broker historical bar volume is in lots of 100" could only be filed in one trading
        project (invisible to the other five that will hit it) or in ``global`` (noise in every
        creative project's brief). Neither is right, so it stayed siloed.
        """
        return f"cluster_{name}"

    # --- parsing a caller-supplied scope ------------------------------------------------------
    #
    # The inverse of the constructors above. Callers (MCP tool args, HTTP query/body fields) send a
    # loose string; these turn it into the ids the engines take. It lives here because three
    # DIVERGENT copies had grown — `mcp/tools.py` understood clusters, `api/routes/knowledge.py`
    # only stripped a `project_` prefix, and `api/routes/search.py` prepended `project_` to whatever
    # it was given. All three shapes look reasonable in isolation and two of them silently mangled a
    # cluster scope: `cluster:trading` became the project id `cluster:trading`, hence the group_id
    # `project_cluster:trading`, which Graphiti rejects for the colon on write (2026-07-27) and which
    # simply matches nothing on read. One parser, one behaviour, no drift.
    #
    # Accepted forms (case-insensitive on the prefix, and the bare form is a project id):
    #   None              -> the caller's default project, if it has one
    #   "global"          -> global scope
    #   "cluster:trading" / "cluster_trading"        -> the trading cluster
    #   "project:acme-api" / "project_acme-api" / "acme-api" -> that project

    _CLUSTER_PREFIXES = ("cluster:", "cluster_")
    _PROJECT_PREFIXES = ("project:", "project_")

    @staticmethod
    def parse_request(
        scope: str | None, *, default_project: str | None = None
    ) -> tuple[str | None, str | None]:
        """Resolve a caller-supplied scope to ``(cluster, project_id)`` for the write path.

        At most one is ever non-None; ``(None, None)`` means global. Note the asymmetry in what
        ``None`` means, which is deliberate and why ``default_project`` is a parameter rather than
        baked in: an MCP server runs inside one project and should default to it, while an HTTP
        caller has no such context and defaults to global.
        """
        if scope is None:
            return None, default_project
        text = scope.strip()
        lowered = text.lower()
        if not text or lowered == GLOBAL_SCOPE:
            return None, None
        for prefix in Scope._CLUSTER_PREFIXES:
            if lowered.startswith(prefix):
                # Cluster names are lowercase by convention (`cluster_trading`).
                return lowered[len(prefix):] or None, None
        for prefix in Scope._PROJECT_PREFIXES:
            if lowered.startswith(prefix):
                return None, text[len(prefix):] or None
        return None, text

    @staticmethod
    def group_id_for_request(scope: str | None) -> str | None:
        """The single ``group_id`` a scope string names, or None meaning "search every scope".

        For read paths that query group_ids directly (``search``). Unlike ``parse_request`` this
        does not need a default, because "no scope given" means unrestricted rather than "mine".
        """
        if scope is None or not scope.strip():
            return None
        cluster, project_id = Scope.parse_request(scope)
        if cluster:
            return Scope.cluster(cluster)
        if project_id:
            return Scope.project(project_id)
        return GLOBAL_SCOPE

    @staticmethod
    def compose(
        project_id: str | None = None,
        agent_role: str | None = None,
        cluster: str | None = None,
    ) -> list[str]:
        """Build the ordered group_id list for a retrieval call.

        Always includes ``global``; adds ``cluster_C``, ``project_X`` and ``agent_Y`` when given.
        Ordered widest-to-narrowest so the list reads as the scope lattice::

            Scope.compose("acme-api", cluster="trading")
            -> ["global", "cluster_trading", "project_acme-api"]
        """
        scopes = [GLOBAL_SCOPE]
        if cluster:
            scopes.append(Scope.cluster(cluster))
        if project_id:
            scopes.append(Scope.project(project_id))
        if agent_role:
            scopes.append(Scope.agent(agent_role))
        return scopes


# ──────────────────────────────────────────────────────────────────────────────
# Entity types (plan Part 3 "Core Entity Types")
# Built-in Graphiti fields (uuid/name/summary/group_id/created_at) are omitted.
# ──────────────────────────────────────────────────────────────────────────────


class Project(BaseModel):
    """A software project tracked by Synapse (e.g. acme-store, acme-docs, acme-api)."""

    tech_stack: str | None = Field(
        None, description="Primary technologies, languages, and frameworks the project uses."
    )
    status: Literal["active", "paused", "archived"] | None = Field(
        None, description="Lifecycle status: active, paused, or archived."
    )


class Decision(BaseModel):
    """A deliberate choice that was made, with its reasoning. The 'why' behind the work."""

    rationale: str | None = Field(
        None, description="WHY this decision was made — the reasoning and trade-offs."
    )
    alternatives_considered: str | None = Field(
        None, description="Options that were weighed and rejected, and briefly why."
    )
    confidence: Literal["tentative", "settled", "locked"] | None = Field(
        None, description="How settled this is: tentative, settled, or locked."
    )
    made_by: str | None = Field(
        None, description="Who made the decision — a human or a specific agent/role."
    )


class Convention(BaseModel):
    """An established 'always do it this way' rule for a project or globally."""

    category: Literal["code_style", "architecture", "naming", "workflow", "testing"] | None = Field(
        None, description="Kind of convention: code_style, architecture, naming, workflow, or testing."
    )
    example: str | None = Field(
        None, description="A concrete code snippet or illustration of the convention."
    )


class Lesson(BaseModel):
    """Something learned the hard way — a gotcha, failure, anti-pattern, or best practice."""

    context: str | None = Field(
        None, description="The situation or task that revealed this lesson."
    )
    lesson_type: Literal["gotcha", "best_practice", "anti_pattern", "failure"] | None = Field(
        None, description="Type: gotcha, best_practice, anti_pattern, or failure."
    )
    severity: Literal["low", "medium", "high", "critical"] | None = Field(
        None, description="How important/costly this lesson is: low, medium, high, or critical."
    )
    source_project: str | None = Field(
        None, description="The project where this lesson was discovered."
    )


class Research(BaseModel):
    """A concluded investigation with findings worth keeping."""

    findings: str | None = Field(
        None, description="The key conclusions reached by the research."
    )
    sources: str | None = Field(
        None, description="URLs, papers, or references the findings are based on."
    )
    relevance_tags: str | None = Field(
        None, description="Topics/keywords this research is relevant to."
    )


class Pattern(BaseModel):
    """A reusable solution shape that applies across projects (usually global scope)."""

    problem: str | None = Field(
        None, description="The recurring problem this pattern solves."
    )
    solution: str | None = Field(
        None, description="How the pattern solves it."
    )
    used_in: str | None = Field(
        None, description="Projects that use this pattern (drives cross-project links)."
    )


class Tool(BaseModel):
    """A library, framework, service, or tool that was chosen and why."""

    category: str | None = Field(
        None, description="What kind of tool — e.g. database, framework, CI, vector store."
    )
    purpose: str | None = Field(
        None, description="What the tool is used for in the project(s)."
    )
    chosen_over: str | None = Field(
        None, description="Alternatives that were rejected in favor of this tool."
    )
    rationale: str | None = Field(
        None, description="Why this tool was selected."
    )


class Runbook(BaseModel):
    """An ordered, executable procedure — "how to do X here" (research §3, roadmap item 18).

    The third memory type. `Decision`/`Convention`/`Lesson` are *semantic* — they assert what is
    true. A Runbook is *procedural*: its value is the ORDER of its steps, and a procedure whose
    steps have been reordered or merged is not a degraded procedure, it is a wrong one.

    That distinction is why this type is **deliberately absent from `ENTITY_TYPES`**, unlike every
    other class here. `ENTITY_TYPES` is handed to Graphiti's LLM extractor, and extraction is
    precisely what destroys step order — measured on the live graph before building this:

    * `Acme-Jobs TDD workflow: failing test → implementation → integration → commit` — a
      Convention whose entire sequence had been crammed into the node **name**, the only field
      that survives extraction intact.
    * `Acme-Sim 10-item live-eligibility checklist` and `13-point pre-publish compliance
      checklist` — only the **count** survived. Not one of the 23 items is in the graph.
    * `docs/runbook/gateway.md` — an untyped node that is a **pointer to a file** the graph cannot
      read.

    Letting the extractor mint Runbooks would therefore produce Runbooks with no steps: the
    current broken state, wearing a label that claims otherwise. A Runbook exists ONLY via the
    structured write path (`RunbookStore.upsert`), which stores `steps` as a node property and
    never asks an LLM to preserve them.

    ``verified_at`` is the field that keeps a runbook honest. Procedural knowledge rots faster
    than semantic knowledge — a deploy sequence silently stops working when the tooling moves —
    and unlike a stale fact, a stale procedure fails *while you are following it*.
    """

    purpose: str | None = Field(
        None, description="What this procedure accomplishes, and when to reach for it."
    )
    steps: list[str] = Field(
        default_factory=list,
        description="The ordered steps. Order is the payload — never reorder or deduplicate.",
    )
    verified_at: datetime | None = Field(
        None, description="When these steps were last confirmed to actually work."
    )
    prerequisites: str | None = Field(
        None, description="What must already be true before step 1 (access, services, state)."
    )


RUNBOOK_LABEL = "Runbook"


# Anything that doesn't fit the above is left to Graphiti's default `Entity` label
# (the generic ENTITY in the plan) — no custom class needed.

# Handed to Graphiti's extractor. `Runbook` is intentionally NOT here — see its docstring: the
# extractor cannot preserve step order, so an extracted Runbook would be an empty one.
ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Project": Project,
    "Decision": Decision,
    "Convention": Convention,
    "Lesson": Lesson,
    "Research": Research,
    "Pattern": Pattern,
    "Tool": Tool,
}


# ──────────────────────────────────────────────────────────────────────────────
# Edge types (plan Part 3 "Relationship Types")
# Docstrings guide the extractor on when to draw each edge. Most carry no extra
# attributes; a few hold meaningful metadata.
# ──────────────────────────────────────────────────────────────────────────────


class Supersedes(BaseModel):
    """An earlier decision is replaced by a later one (temporal evolution)."""

    reason: str | None = Field(None, description="Why the newer decision replaced the older one.")


class AppliesTo(BaseModel):
    """A decision applies to a specific project."""


class DerivedFrom(BaseModel):
    """A convention was derived from a lesson learned."""


class DiscoveredIn(BaseModel):
    """A lesson was discovered while working in a project."""


class Informed(BaseModel):
    """Research informed a decision."""


class UsedIn(BaseModel):
    """A pattern is used in a project."""


class HasGotcha(BaseModel):
    """A tool has an associated gotcha/lesson."""


class Contradicts(BaseModel):
    """Two decisions conflict — flagged for human resolution."""

    resolution_status: Literal["open", "resolved"] | None = Field(
        None, description="Whether the contradiction has been resolved."
    )


class RelatedTo(BaseModel):
    """A general association between two conventions (or other knowledge)."""


class References(BaseModel):
    """One research item references another."""


class SharesPattern(BaseModel):
    """Two projects share a common pattern — the cross-project magic link."""


# ── Types added 2026-07-25 (research §2.2, roadmap item 24) ───────────────────
# The original 11 types covered 1,664 of 3,018 edges. The extractor expressed the rest as 516
# invented names — but the *recurring* tail was real vocabulary the schema simply did not model,
# and flattening it into RelatedTo would have destroyed meaning. These 12 types are that tail,
# grouped by relation and DIRECTION (each docstring is read by the extractor, so it states the
# direction explicitly). Together they type ~432 previously-untypeable edges.


class Uses(BaseModel):
    """The source depends on / is built with the target. Direction: user -> dependency.

    Use for "Acme-API uses TimescaleDB", "the decision uses library X", "built with Spring Boot".
    This is the INVERSE of ``UsedIn`` — do not confuse them: ``UsedIn`` runs pattern -> project.
    """


class Implements(BaseModel):
    """The source provides a concrete implementation of the target. Direction: implementer -> spec.

    Use for "the gateway service implements the retry pattern", "Tool X implements the research
    finding". For "X is defined/located in Y" use ``DefinedIn`` instead.
    """


class DefinedIn(BaseModel):
    """The source is defined, implemented or located inside the target. Direction: thing -> place.

    Use for "select_pead_candidates is defined in pead_universe.py", "the convention is
    implemented in the gateway module".
    """


class PartOf(BaseModel):
    """The source is a component or member of the target. Direction: part -> whole.

    Use for "the watchtower service is part of Acme-API", "this belongs to the trading stack".
    """


class Contains(BaseModel):
    """The source contains or includes the target. Direction: whole -> part.

    The inverse of ``PartOf``; use whichever matches the sentence's subject.
    """


class Causes(BaseModel):
    """The source brought about the target. Direction: cause -> effect.

    Use for "the duplicate lot created the position mismatch". For the reverse phrasing
    ("the mismatch was caused by ...") use ``CausedBy`` — both exist so neither needs the
    endpoints swapped.
    """


class CausedBy(BaseModel):
    """The source was brought about by the target. Direction: effect -> cause."""


class Fixes(BaseModel):
    """The source resolves the target problem, bug or lesson. Direction: remedy -> problem.

    Use for "the debounce change fixes the gapping-orders lesson".
    """


class Enforces(BaseModel):
    """The source makes the target mandatory, or gates on it. Direction: enforcer -> rule.

    Use for "the CI gate enforces the BigDecimal convention", "the verdict table gates the
    volfloor rollout".
    """


class Affects(BaseModel):
    """The source has an impact on the target, without causing or fixing it outright.

    Direction: influencer -> influenced. The weakest of the causal family — prefer ``Causes``,
    ``CausedBy`` or ``Fixes`` when the relation is definite.
    """


class ContributesTo(BaseModel):
    """The source is one of several contributing factors to the target.

    Direction: factor -> outcome. Use when the target has multiple causes and this is one.
    """


class Calls(BaseModel):
    """The source invokes the target at runtime. Direction: caller -> callee.

    Code-level relation: "sweepOrphanOrdersPeriodic calls cancelOrder".
    """


EDGE_TYPES: dict[str, type[BaseModel]] = {
    "Supersedes": Supersedes,
    "AppliesTo": AppliesTo,
    "DerivedFrom": DerivedFrom,
    "DiscoveredIn": DiscoveredIn,
    "Informed": Informed,
    "UsedIn": UsedIn,
    "HasGotcha": HasGotcha,
    "Contradicts": Contradicts,
    "RelatedTo": RelatedTo,
    "References": References,
    "SharesPattern": SharesPattern,
    # Added 2026-07-25 (roadmap item 24) — the recurring residual, now first-class.
    "Uses": Uses,
    "Implements": Implements,
    "DefinedIn": DefinedIn,
    "PartOf": PartOf,
    "Contains": Contains,
    "Causes": Causes,
    "CausedBy": CausedBy,
    "Fixes": Fixes,
    "Enforces": Enforces,
    "Affects": Affects,
    "ContributesTo": ContributesTo,
    "Calls": Calls,
}


# Which edge types are allowed between which entity labels. Graphiti uses this to
# constrain extraction. ("Entity", "Entity") is the catch-all fallback pair.
EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Decision", "Decision"): ["Supersedes", "Contradicts"],
    ("Decision", "Project"): ["AppliesTo"],
    ("Convention", "Lesson"): ["DerivedFrom"],
    ("Convention", "Convention"): ["RelatedTo"],
    ("Lesson", "Project"): ["DiscoveredIn"],
    ("Research", "Decision"): ["Informed"],
    ("Research", "Research"): ["References"],
    ("Pattern", "Project"): ["UsedIn"],
    ("Project", "Project"): ["SharesPattern", "PartOf"],
    # Pairs below were derived from the live graph's actual endpoint labels per relation
    # (roadmap item 24), so the map permits what the extractor demonstrably wants to say.
    ("Project", "Tool"): ["Uses"],
    ("Decision", "Tool"): ["Uses", "Contains"],
    ("Project", "Convention"): ["Uses", "Enforces"],
    ("Project", "Pattern"): ["Uses", "Implements"],
    ("Tool", "Pattern"): ["Implements"],
    ("Research", "Tool"): ["Uses"],
    ("Convention", "Pattern"): ["Enforces"],
    ("Convention", "Project"): ["Enforces"],
    ("Decision", "Convention"): ["Enforces"],
    ("Convention", "Entity"): ["DefinedIn", "Calls"],
    ("Pattern", "Entity"): ["DefinedIn"],
    ("Pattern", "Tool"): ["DefinedIn", "Uses"],
    ("Pattern", "Lesson"): ["Causes"],
    ("Lesson", "Entity"): ["CausedBy", "Affects"],
    ("Lesson", "Convention"): ["CausedBy"],
    ("Lesson", "Pattern"): ["Affects"],
    ("Decision", "Entity"): ["Fixes", "Contains", "Calls"],
    ("Decision", "Lesson"): ["Fixes"],
    ("Decision", "Pattern"): ["Fixes", "Implements"],
    ("Entity", "Project"): ["PartOf", "ContributesTo"],
    ("Entity", "Tool"): ["Contains", "Uses"],
    ("Entity", "Lesson"): ["ContributesTo", "Causes"],
    ("Entity", "Research"): ["PartOf"],
    ("Tool", "Lesson"): ["ContributesTo", "HasGotcha"],
    ("Tool", "Tool"): ["Contains", "Uses"],
    ("Tool", "Research"): ["Implements"],
    ("Tool", "Decision"): ["Implements"],
    ("Research", "Entity"): ["Affects"],
    ("Entity", "Entity"): ["RelatedTo", "Calls", "DefinedIn", "PartOf", "Contains", "Causes"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Edge-name canonicalization (research §2.2)
# ──────────────────────────────────────────────────────────────────────────────
# Measured on the live graph 2026-07-25: **535 distinct edge names over 3,018 edges.** The 11
# schema types above cover 1,664; the other 516 names are extractor inventions covering 1,354
# edges — and **328 of those names appear on exactly ONE edge** (PINNED_SECTOR_ETF,
# DETECTED_UNTRACKED, IS_IMPLEMENTED_SERVICE_OF ...). That is not a vocabulary, it is noise: it
# makes typed traversal impossible and blocks any structural retrieval lens.
#
# Canonicalization is deliberately CONSERVATIVE. Only names that mean the same relation *in the
# same direction* are folded in. Two traps we do not fall into:
#
#   * `USES` (134 edges) is NOT folded into `UsedIn`. The schema's UsedIn runs Pattern->Project
#     ("a pattern is used in a project"); `USES` runs the other way (project->tool). Folding them
#     would silently invert 134 edges.
#   * Genuinely different relations (`CAUSED_BY`, `FIXES`, `ENFORCES`, `CONTAINS`, `PART_OF`) are
#     NOT flattened into `RelatedTo`. Collapsing a specific relation into a generic one destroys
#     meaning — the same mistake as merging two facts that share a sentence frame. Those belong in
#     a deliberate schema EXTENSION, which the residual report exists to inform.
#
# Keys are normalized (lowercase, alphanumerics only) so casing and separators are handled for free.
EDGE_NAME_SYNONYMS: dict[str, str] = {
    # AppliesTo — tense/preposition variants, all decision->project
    "appliedto": "AppliesTo",
    "appliedin": "AppliesTo",
    "applicableto": "AppliesTo",
    # UsedIn — only the passive forms, never the active "USES" (see above)
    "isusedin": "UsedIn",
    # DerivedFrom
    "derivesfrom": "DerivedFrom",
    "derivedfor": "DerivedFrom",
    # References
    "referenced": "References",
    "cites": "References",
    # Contradicts
    "conflictswith": "Contradicts",
    "conflicts": "Contradicts",
    # Informed
    "informs": "Informed",
    # SharesPattern
    "sharedpattern": "SharesPattern",
    # HasGotcha
    "hasgotchas": "HasGotcha",
    # --- folded onto the types added 2026-07-25 (roadmap item 24) ---
    # Uses: "built with"/"built on"/"uses tool" all state a dependency, same direction.
    "usestool": "Uses",
    "builtwith": "Uses",
    "builton": "Uses",
    # Implements: past tense is the same relation.
    "implemented": "Implements",
    # DefinedIn: implemented-in / located-in are the same containment-of-a-definition.
    "implementedin": "DefinedIn",
    "locatedin": "DefinedIn",
    # PartOf: membership, however phrased.
    "belongsto": "PartOf",
    "isimplementedserviceof": "PartOf",
    # Contains
    "includes": "Contains",
    # Enforces: a gate is an enforcement mechanism.
    "gates": "Enforces",
    # Causes: past tense. Note CAUSED_BY is NOT folded here — it is the inverse direction and has
    # its own type, so neither spelling needs its endpoints swapped.
    "caused": "Causes",
    # Affects
    "affected": "Affects",
    # ContributesTo: tense variant only.
    "contributedto": "ContributesTo",
}


def _normalize_edge_name(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


# Canonical names keyed by their normalized form — handles APPLIES_TO/applies_to/AppliesTo.
_CANONICAL_BY_NORM: dict[str, str] = {_normalize_edge_name(k): k for k in EDGE_TYPES}


def canonical_edge_name(name: str) -> str:
    """Fold an edge name onto its schema type, or return it unchanged.

    Returns *name* untouched when there is no confident, direction-preserving mapping — an unknown
    relation keeps its own name rather than being flattened into a generic one.
    """
    if not name:
        return name
    if name in EDGE_TYPES:
        return name
    key = _normalize_edge_name(name)
    return _CANONICAL_BY_NORM.get(key) or EDGE_NAME_SYNONYMS.get(key) or name


__all__ = [
    "GLOBAL_SCOPE",
    "Scope",
    "EDGE_NAME_SYNONYMS",
    "canonical_edge_name",
    "Project",
    "Decision",
    "Convention",
    "Lesson",
    "Research",
    "Pattern",
    "Tool",
    "ENTITY_TYPES",
    "EDGE_TYPES",
    "EDGE_TYPE_MAP",
]
