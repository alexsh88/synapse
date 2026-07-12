"""Evaluation case definitions for Synapse retrieval quality.

Each case asserts that a query surfaces a fact containing ANY of ``expect_any``
(case-insensitive substrings) within the top ``k``, optionally from ``expect_scope``.
The public ``GOLDEN_SET`` ships as ``DEMO_CASES`` (fictional acme-* projects) so the
repository is safe to share. When ``synapse/eval/cases_private.py`` is present (not
committed; see .gitignore) its ``GOLDEN_SET`` replaces this one automatically.

Matching rules
--------------
* ``expect_any``  — a result is a hit when ANY of these substrings appear in the
  fact text (case-insensitive).  Legacy; always supported.
* ``keywords``    — when provided, ALL keywords must appear in the fact text
  (case-insensitive AND-match).  Reduces paraphrase brittleness.
  Either ``expect_any`` OR ``keywords`` must produce a hit; the two lists are
  checked independently and either one winning makes the result a hit.
* ``must_not_match`` — substrings that must NOT appear in any top-k fact.
  Each occurrence is counted as a *violation* (independent of the hit score).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    query: str
    category: Literal["acme-api", "acme-data", "global", "cross_project", "negative",
                      "acme-api", "acme-web", "acme-infra"]
    expect_any: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    must_not_match: list[str] = Field(default_factory=list)
    mode: Literal["recall", "search"] = "recall"
    project_id: str | None = None          # recall scope (the seat the agent queries from)
    expect_scope: str | None = None         # require the hit to come from this scope
    k: int = Field(default=5)


# ---------------------------------------------------------------------------
# DEMO_CASES — fictional acme-* projects, safe to commit
# ---------------------------------------------------------------------------

DEMO_CASES: list[EvalCase] = [
    # --- acme-api: Python FastAPI service decisions / conventions ---
    EvalCase(id="api-pydantic", category="acme-api", project_id="acme-api",
             query="what validation library does acme-api use for request/response schemas?",
             expect_any=["pydantic"]),
    EvalCase(id="api-auth", category="acme-api", project_id="acme-api",
             query="how is authentication handled in the acme-api service?",
             expect_any=["jwt", "bearer", "oauth"]),
    EvalCase(id="api-db-driver", category="acme-api", project_id="acme-api",
             query="which PostgreSQL driver does acme-api use for async queries?",
             expect_any=["asyncpg", "sqlalchemy", "databases"]),
    EvalCase(id="api-error-handling", category="acme-api", project_id="acme-api",
             query="what is the convention for returning error responses in acme-api?",
             expect_any=["422", "rfc 7807", "problem+json", "detail"]),

    # --- acme-web: React storefront conventions ---
    EvalCase(id="web-state", category="acme-web", project_id="acme-web",
             query="what state management library does acme-web use for server data?",
             expect_any=["tanstack", "react query", "swr"]),
    EvalCase(id="web-styling", category="acme-web", project_id="acme-web",
             query="how is CSS handled in acme-web?",
             expect_any=["tailwind", "css modules", "styled-components"]),
    EvalCase(id="web-routing", category="acme-web", project_id="acme-web",
             query="what router does acme-web use for client-side navigation?",
             expect_any=["react router", "tanstack router", "next"]),

    # --- acme-infra: Terraform / Kubernetes tooling ---
    EvalCase(id="infra-cloud", category="acme-infra", project_id="acme-infra",
             query="which cloud provider does acme-infra target?",
             expect_any=["aws", "gcp", "azure", "eks", "gke", "aks"]),
    EvalCase(id="infra-cd", category="acme-infra", project_id="acme-infra",
             query="what continuous delivery tool manages deployments in acme-infra?",
             expect_any=["argocd", "flux", "helm", "argo"]),

    # --- global: cross-cutting conventions ---
    EvalCase(id="glob-commit", category="global", project_id="acme-api",
             query="what commit message convention is used across acme projects?",
             expect_any=["conventional commits", "feat:", "fix:", "chore:"],
             expect_scope="global"),

    # --- NEGATIVE / DISTRACTOR CASES ---

    # 1. Cross-project leakage: acme-api (Python/FastAPI) must not surface
    #    infrastructure concepts (Terraform, Kubernetes) when queried for API patterns.
    EvalCase(
        id="neg-api-infra-leakage",
        category="negative",
        project_id="acme-api",
        query="how do I define a new FastAPI route handler with dependency injection?",
        expect_any=[],
        must_not_match=["terraform", "kubectl", "helm chart", "kubernetes", "argocd"],
        keywords=[],
    ),

    # 2. Cross-project leakage reverse: acme-infra must not surface React/frontend
    #    concepts when queried for infrastructure patterns.
    EvalCase(
        id="neg-infra-web-leakage",
        category="negative",
        project_id="acme-infra",
        query="how do I add a new Kubernetes namespace and apply resource quotas?",
        expect_any=[],
        must_not_match=["react", "tailwind", "vite", "npm", "useState", "component"],
        keywords=[],
    ),

    # 3. Off-topic query — completely unrelated domain. Nothing confident should surface.
    EvalCase(
        id="neg-off-topic-cooking",
        category="negative",
        project_id="acme-api",
        query="what is the best way to bake sourdough bread at home?",
        expect_any=[],
        must_not_match=["pydantic", "fastapi", "postgresql", "jwt", "tailwind",
                        "terraform", "kubernetes"],
    ),

    # 4. Off-topic query from web seat — gardening is irrelevant to any acme project.
    EvalCase(
        id="neg-off-topic-gardening",
        category="negative",
        project_id="acme-web",
        query="how do I grow tomatoes in a container garden?",
        expect_any=[],
        must_not_match=["react", "tailwind", "fastapi", "postgres", "terraform"],
    ),

    # 5. Superseded knowledge: an old REST convention was replaced.
    #    The new convention must surface; the old one must not appear as authoritative.
    EvalCase(
        id="neg-superseded-rest-version",
        category="negative",
        project_id="acme-api",
        query="what versioning strategy does acme-api use for its REST endpoints?",
        expect_any=["v1", "/api/v1", "url versioning"],
        must_not_match=["header versioning is the convention",
                        "use accept-version header"],
    ),
]

# ---------------------------------------------------------------------------
# Active golden set — replaced by cases_private.GOLDEN_SET when present
# ---------------------------------------------------------------------------

try:
    from synapse.eval.cases_private import GOLDEN_SET  # noqa: F401
except ImportError:
    GOLDEN_SET = DEMO_CASES
