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
    def compose(project_id: str | None = None, agent_role: str | None = None) -> list[str]:
        """Build the ordered group_id list for a retrieval call.

        Always includes ``global``; adds ``project_X`` and ``agent_Y`` when given.
        e.g. ``Scope.compose("acme-store")`` -> ``["global", "project_acme-store"]``.
        """
        scopes = [GLOBAL_SCOPE]
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


# Anything that doesn't fit the above is left to Graphiti's default `Entity` label
# (the generic ENTITY in the plan) — no custom class needed.

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
    ("Tool", "Lesson"): ["HasGotcha"],
    ("Project", "Project"): ["SharesPattern"],
    ("Entity", "Entity"): ["RelatedTo"],
}


__all__ = [
    "GLOBAL_SCOPE",
    "Scope",
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
