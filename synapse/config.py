"""Central configuration for Synapse.

Connection settings load from environment / `.env` via pydantic-settings.
Import `settings` anywhere: `from synapse.config import settings`.

Host ports are remapped (see docs/readiness-check.md §2b) when the standard
ports conflict with other stacks running on the same machine.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the repo-root .env, so the MCP server finds it even when launched
# from another project's working directory (synapse/config.py -> repo root).
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Neo4j (Graphiti graph backend) ---
    # 127.0.0.1 (not "localhost") forces IPv4 — Docker's port-forward serves IPv4 reliably,
    # while the driver resolving localhost to IPv6 ::1 causes intermittent defunct connections.
    neo4j_uri: str = "bolt://127.0.0.1:7688"
    neo4j_http: str = "http://127.0.0.1:7475"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # --- Redis (Celery broker + brief cache) ---
    redis_url: str = "redis://localhost:6382/0"

    # --- API ---
    synapse_api_url: str = "http://localhost:8848"
    synapse_api_port: int = 8848

    # --- Optional API key (empty string = auth disabled, current dev behavior preserved) ---
    # When set, all API routes require X-Synapse-Key: <value> header; returns 401 otherwise.
    api_key: str = ""

    # --- LLM (provider decisions: see Phase-1A research §6) ---
    anthropic_api_key: str = ""
    # Sonnet does the quality-critical graph extraction (via Graphiti).
    extraction_model: str = "claude-sonnet-4-6"
    # Haiku does the cheap, high-volume triage (worth-storing? type? global?).
    triage_model: str = "claude-haiku-4-5"

    # --- Extraction routing (cost; see docs/research/local-extraction-models.md) ---
    # cloud  = Claude Sonnet 4.6 (default; the locked-decision quality baseline).
    # local  = Gemma3-12B via Ollama, strict json_schema ($0; ~81%/64% of Sonnet entities/edges,
    #          ~14% of dense writes silently fail). hybrid = local first, Sonnet fallback on failure.
    extraction_mode: Literal["cloud", "hybrid", "local"] = "cloud"
    local_extraction_model: str = "gemma3:12b"   # measured best quality/GB at 16GB VRAM

    # --- Session-lesson auto-capture (docs/plans/2026-06-06-session-lesson-capture-design.md) ---
    # PreCompact/SessionEnd hooks feed transcripts to a Haiku judge; confidence >= threshold auto-stores,
    # the rest queue for review. Three R2 gates: strict judge + threshold + write-pipeline triage/dedup.
    capture_enabled: bool = True
    capture_autostore_threshold: float = 0.8
    capture_model: str = "claude-haiku-4-5"

    # --- Write pipeline thresholds (plan Part 4) ---
    dedup_threshold: float = 0.9   # >= this cosine sim => duplicate, don't re-store
    relate_floor: float = 0.75     # >= this (but < dedup) => adjudicate
    # Top-k adjudication (roadmap item 16). Judging only the single nearest fact meant a write
    # contradicting the SECOND-nearest was never flagged: the live graph held 7 Contradicts
    # edges against 3,039 facts. Fetching neighbours is one cheap ANN query while each
    # adjudication is an LLM call, so fetch width and judgement budget are separate knobs.
    adjudication_candidates: int = 5
    max_adjudications: int = 3

    # --- Curation fact<->fact thresholds (Phase 10/11; docs/architecture/curation.md) ---
    # Higher than the write-time dedup_threshold: measured on the live ~589-node graph,
    # fact<->fact at 0.90 over-merges distinct facts that share sentence structure/topic
    # ("X computes Liquidity Sweeps" vs "X computes Order Blocks" score ~0.90). 0.97 = high precision.
    curation_dedup_threshold: float = 0.97  # >= => merge candidate (Curate panel)
    curation_review_floor: float = 0.90     # [floor, dedup) => "possibly related, human glance"
    curation_pair_limit: int = 500          # hard cap on the O(n^2) pair scan; logged when hit

    # --- Graphiti concurrency (caps simultaneous LLM/embedding calls; see knowledge_engine) ---
    # Mild cap. NOTE: local bge-m3 on Ollama has a separate NaN-under-sustained-load
    # instability (see docs/architecture/mcp-server.md "Known issue: embedder NaN");
    # concurrency is not the root cause, so this is just light defense.
    graphiti_max_coroutines: int = 5

    # --- Embedder endpoint (Ollama, OpenAI-compatible) ---
    # Host default. The dockerized API overrides via OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
    # so the container reaches the host's GPU Ollama (which keeps OLLAMA_FLASH_ATTENTION=0).
    ollama_base_url: str = "http://127.0.0.1:11434/v1"

    # --- Filesystem ---
    # Root under which connected projects live. Set via PROJECTS_ROOT env var or .env.
    # The API container overrides via PROJECTS_ROOT=/projects (the mounted host projects dir).
    # Used by the project connector (F2) to write wiring files and read docs for seeding.
    # Empty string = not configured; registry/connector will skip path-dependent operations.
    projects_root: str = ""
    # Additional roots to search for a project folder, comma-separated (EXTRA_PROJECT_ROOTS).
    # Not every project lives under one parent directory, and the container can only reach a host
    # directory that is bind-mounted — so an out-of-root project is a ROOT of its own, not a
    # subdirectory. Mount each one and list its container-side parent here. The primary root above
    # is still searched first and still owns the connected-projects overlay.
    extra_project_roots: str = ""
    # The HOST path of this synapse repo — written verbatim into project .mcp.json / hook commands
    # so Claude Code (running on the host) invokes the right venv python + scripts. Stays a host
    # path even when the API runs in a container (where the code is mounted at /app).
    # Empty string = not configured; the connector then falls back to this repo's own on-disk
    # location, which is the correct host path when wiring is run from the host (not the container).
    synapse_host_dir: str = ""

    # --- Project registry ---
    # Path to the JSON file listing connected projects (see projects.example.json).
    # Defaults to projects.json at the repo root; falls back to projects.example.json
    # with a logged warning when the primary file is absent.
    projects_file: str = str(Path(__file__).resolve().parents[1] / "projects.json")

    def require_neo4j_password(self) -> str:
        """Return the Neo4j password, raising ValueError if it is empty.

        Call this in any code path that actually connects to Neo4j so that
        missing-password errors surface early with a clear message rather than
        surfacing as a cryptic authentication failure.
        """
        if not self.neo4j_password:
            raise ValueError(
                "NEO4J_PASSWORD is not set. "
                "Add it to your .env file or set the NEO4J_PASSWORD environment variable."
            )
        return self.neo4j_password


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
