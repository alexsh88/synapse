"""Curation engine (Phase 10) — keep the brain healthy, never lose knowledge.

Analysis is read-only; mutations are reversible and backup-first. NOTHING here
hard-deletes. See ``docs/architecture/curation.md`` for the safety contract (R8/R4/R3).

Injected ``graphiti`` (driver) + ``BackupService`` so it's unit-testable with a fake
driver and no live Neo4j (CLAUDE.md §4).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from synapse.config import settings
from synapse.core.backup import BackupService, CurationSafetyError
from synapse.core.vector_index import FACT_VECTOR_INDEX

logger = logging.getLogger("synapse.curation")

# Native relationship vector index over RELATES_TO.fact_embedding, created (idempotently) by the write
# pipeline (WP-B, synapse/core/write_pipeline.py). Candidate generation for fact<->fact dedup/review is a
# per-edge k-NN over this index instead of the old O(n^2) cross-join (WP-H item 1).
# Imported from synapse.core.vector_index so both modules always reference the same literal.
_FACT_VECTOR_INDEX = FACT_VECTOR_INDEX
# Neighbors fetched per edge. Small: dedup/review only needs an edge's closest few facts; the band filter
# then keeps whichever land in [floor, ceil). 8 covers realistic near-duplicate fan-out with headroom.
_KNN_PER_EDGE = 8


# ──────────────────────────────────────────────────────────────────────────────
# Suggestion + result models
# ──────────────────────────────────────────────────────────────────────────────


class FactRef(BaseModel):
    uuid: str
    fact: str


class DuplicateCluster(BaseModel):
    scope: str
    canonical: FactRef           # earliest-created fact — the one to keep
    duplicates: list[FactRef]    # merge candidates (superseded into canonical)
    max_similarity: float


class StaleItem(BaseModel):
    uuid: str
    fact: str
    scope: str
    created_at: datetime | None = None
    age_days: int | None = None


class ReviewPair(BaseModel):
    """Gray-band similar pair — a POSSIBLE overlap/contradiction for a human glance."""

    scope: str
    a: FactRef
    b: FactRef
    similarity: float


class CurationSuggestions(BaseModel):
    duplicates: list[DuplicateCluster] = Field(default_factory=list)
    stale: list[StaleItem] = Field(default_factory=list)
    review_pairs: list[ReviewPair] = Field(default_factory=list)
    generated_at: datetime | None = None


class ApplyResult(BaseModel):
    ok: bool
    action: str
    edge_uuid: str
    backup_path: str | None = None
    detail: str = ""


def _native(value):
    return value.to_native() if hasattr(value, "to_native") else value


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────


class CurationEngine:
    def __init__(
        self,
        graphiti,
        backup: BackupService,
        *,
        dedup_threshold: float | None = None,
        review_floor: float | None = None,
        pair_limit: int | None = None,
        now: datetime | None = None,
    ) -> None:
        self._driver = graphiti.driver
        self.backup = backup
        # Fact<->fact thresholds (distinct from the write pipeline's episode->fact dedup).
        self.dedup_threshold = dedup_threshold if dedup_threshold is not None else settings.curation_dedup_threshold
        self.review_floor = review_floor if review_floor is not None else settings.curation_review_floor
        self.pair_limit = pair_limit if pair_limit is not None else settings.curation_pair_limit
        self._now = now  # injectable for deterministic tests
        self._knn_fallback_logged = False  # log the scan fallback ONCE per engine, not per call

    def _utcnow(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    # --- analysis (read-only) ------------------------------------------------

    async def find_duplicates(self, scopes: list[str] | None = None) -> list[DuplicateCluster]:
        rows = await self._similar_pairs(self.dedup_threshold, None, scopes)
        # Union-find the pairs into clusters.
        created: dict[str, datetime | None] = {}
        facts: dict[str, str] = {}
        scope_of: dict[str, str] = {}
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for r in rows:
            a, b = r["a_uuid"], r["b_uuid"]
            facts[a], facts[b] = r["a_fact"], r["b_fact"]
            created[a], created[b] = _native(r["a_created"]), _native(r["b_created"])
            scope_of[a] = scope_of[b] = r["scope"]
            union(a, b)

        # Max similarity per cluster — computed AFTER all unions so a chained
        # cluster (e1~e2~e3) reports the true max, not an intermediate root's.
        maxsim: dict[str, float] = {}
        for r in rows:
            root = find(r["a_uuid"])
            maxsim[root] = max(maxsim.get(root, 0.0), float(r["sim"]))

        clusters: dict[str, list[str]] = {}
        for uid in facts:
            clusters.setdefault(find(uid), []).append(uid)

        _far_future = datetime.max.replace(tzinfo=timezone.utc)
        out: list[DuplicateCluster] = []
        for root, members in clusters.items():
            members.sort(key=lambda u: (created[u] or _far_future, u))
            canonical, *dups = members
            out.append(DuplicateCluster(
                scope=scope_of[canonical],
                canonical=FactRef(uuid=canonical, fact=facts[canonical]),
                duplicates=[FactRef(uuid=u, fact=facts[u]) for u in dups],
                max_similarity=round(maxsim.get(root, 0.0), 4),
            ))
        out.sort(key=lambda c: c.max_similarity, reverse=True)
        return out

    async def find_review_pairs(self, scopes: list[str] | None = None) -> list[ReviewPair]:
        # A human-glance SAMPLE: the 40 most-similar in-band pairs (ordered by sim DESC).
        rows = await self._similar_pairs(self.review_floor, self.dedup_threshold, scopes, limit=40)
        return [
            ReviewPair(
                scope=r["scope"],
                a=FactRef(uuid=r["a_uuid"], fact=r["a_fact"]),
                b=FactRef(uuid=r["b_uuid"], fact=r["b_fact"]),
                similarity=round(float(r["sim"]), 4),
            )
            for r in rows
        ]

    async def _similar_pairs(self, floor: float, ceil: float | None, scopes: list[str] | None,
                             limit: int | None = None):
        """Candidate similar pairs in the band ``[floor, ceil)`` (ceil None = no ceiling).

        Uses a per-edge k-NN over the native relationship vector index (WP-H item 1): O(edges * k)
        instead of the old O(edges^2) cross-join that silently truncated at the pair cap as the corpus
        grew. Returns the SAME record shape as before (a_uuid/a_fact/a_created/b_uuid/b_fact/b_created/
        scope/sim) so ``find_duplicates`` / ``find_review_pairs`` are unchanged.

        Falls back to the brute-force cosine scan if the index query raises (index still populating,
        unsupported build, etc.), logging ONCE per engine.
        """
        try:
            return await self._similar_pairs_knn(floor, ceil, scopes, limit)
        except Exception as exc:  # noqa: BLE001 — any index-path failure → fall back to the full scan
            if not self._knn_fallback_logged:
                logger.warning(
                    "curation k-NN candidate query failed (%s); falling back to the brute-force cosine "
                    "scan for this engine", str(exc)[:120])
                self._knn_fallback_logged = True
            return await self._similar_pairs_scan(floor, ceil, scopes, limit)

    async def _similar_pairs_knn(self, floor: float, ceil: float | None, scopes: list[str] | None,
                                 limit: int | None = None):
        cap = limit if limit is not None else self.pair_limit
        scope_filter = "AND e1.group_id IN $scopes" if scopes else ""
        band = "AND sim < $ceil" if ceil is not None else ""
        res = await self._driver.execute_query(
            f"""
            MATCH (a:Entity)-[e1:RELATES_TO]->(b:Entity)
            WHERE e1.invalid_at IS NULL AND coalesce(e1.archived, false) = false
              AND e1.fact_embedding IS NOT NULL {scope_filter}
            WITH e1, count(*) AS _edges_scanned
            CALL db.index.vector.queryRelationships($index, $k, e1.fact_embedding)
            YIELD relationship AS e2, score AS sim
            WHERE e2.uuid <> e1.uuid AND e1.uuid < e2.uuid
              AND e2.group_id = e1.group_id
              AND e2.invalid_at IS NULL AND coalesce(e2.archived, false) = false
              AND e2.fact_embedding IS NOT NULL
              AND sim >= $floor {band}
            RETURN e1.uuid AS a_uuid, e1.fact AS a_fact, e1.created_at AS a_created,
                   e2.uuid AS b_uuid, e2.fact AS b_fact, e2.created_at AS b_created,
                   e1.group_id AS scope, sim, _edges_scanned
            ORDER BY sim DESC LIMIT $limit
            """,
            index=_FACT_VECTOR_INDEX, k=_KNN_PER_EDGE,
            floor=floor, ceil=ceil, scopes=scopes or [], limit=cap,
        )
        records = res.records
        edges_scanned = records[0]["_edges_scanned"] if records else 0
        logger.info(
            "curation k-NN candidates: %d edges scanned (k=%d), %d banded pairs "
            "[floor=%.2f, ceil=%s]", edges_scanned, _KNN_PER_EDGE, len(records), floor, ceil)
        if len(records) >= cap:
            logger.warning(
                "curation k-NN candidates hit the %d-pair cap (floor=%.2f, ceil=%s) — results truncated; "
                "raise the limit or the threshold.", cap, floor, ceil)
        return records

    async def _similar_pairs_scan(self, floor: float, ceil: float | None, scopes: list[str] | None,
                                  limit: int | None = None):
        """Brute-force O(n^2) cosine cross-join — the fallback when the k-NN index path is unavailable."""
        cap = limit if limit is not None else self.pair_limit
        scope_filter = "AND e1.group_id IN $scopes" if scopes else ""
        band = "AND sim < $ceil" if ceil is not None else ""
        res = await self._driver.execute_query(
            f"""
            MATCH (a:Entity)-[e1:RELATES_TO]->(b:Entity)
            MATCH (c:Entity)-[e2:RELATES_TO]->(d:Entity)
            WHERE e1.group_id = e2.group_id AND e1.uuid < e2.uuid {scope_filter}
              AND e1.invalid_at IS NULL AND e2.invalid_at IS NULL
              AND coalesce(e1.archived, false) = false AND coalesce(e2.archived, false) = false
              AND e1.fact_embedding IS NOT NULL AND e2.fact_embedding IS NOT NULL
            WITH e1, e2, vector.similarity.cosine(e1.fact_embedding, e2.fact_embedding) AS sim
            WHERE sim >= $floor {band}
            RETURN e1.uuid AS a_uuid, e1.fact AS a_fact, e1.created_at AS a_created,
                   e2.uuid AS b_uuid, e2.fact AS b_fact, e2.created_at AS b_created,
                   e1.group_id AS scope, sim
            ORDER BY sim DESC LIMIT $limit
            """,
            floor=floor, ceil=ceil, scopes=scopes or [], limit=cap,
        )
        records = res.records
        # No silent caps: if we hit the limit, the result is truncated — say so.
        if len(records) >= cap:
            logger.warning(
                "curation pair scan hit the %d-pair cap (floor=%.2f, ceil=%s) — results truncated; "
                "raise the limit or the threshold.", cap, floor, ceil)
        return records

    async def find_stale(self, older_than_days: int = 180, scopes: list[str] | None = None) -> list[StaleItem]:
        cutoff = self._utcnow() - timedelta(days=older_than_days)
        scope_filter = "AND e.group_id IN $scopes" if scopes else ""
        res = await self._driver.execute_query(
            f"""
            MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
            WHERE e.invalid_at IS NULL AND coalesce(e.archived, false) = false
              AND e.created_at IS NOT NULL AND e.created_at < $cutoff {scope_filter}
            RETURN e.uuid AS uuid, e.fact AS fact, e.group_id AS scope, e.created_at AS created_at
            ORDER BY e.created_at ASC LIMIT 50
            """,
            cutoff=cutoff, scopes=scopes or [],
        )
        out = []
        for r in res.records:
            created = _native(r["created_at"])
            age = (self._utcnow() - created).days if created else None
            out.append(StaleItem(uuid=r["uuid"], fact=r["fact"], scope=r["scope"],
                                 created_at=created, age_days=age))
        return out

    async def suggestions(self, scopes: list[str] | None = None) -> CurationSuggestions:
        return CurationSuggestions(
            duplicates=await self.find_duplicates(scopes),
            stale=await self.find_stale(scopes=scopes),
            review_pairs=await self.find_review_pairs(scopes),
            generated_at=self._utcnow(),
        )

    # --- mutations (reversible, backup-first) --------------------------------

    async def merge_duplicate(self, canonical_uuid: str, duplicate_uuid: str) -> ApplyResult:
        """Supersede the duplicate into the canonical fact (temporal end, R4). Reversible."""
        # Scoped one-hop backup around the two edges the merge touches (WP-H item 2): O(neighborhood),
        # not O(corpus). verify_no_loss still catches any loss in that neighborhood.
        snapshot = await self.backup.collect(edge_uuids=[duplicate_uuid, canonical_uuid])
        backup_path = await self.backup.snapshot("merge", edge_uuids=[duplicate_uuid, canonical_uuid])
        res = await self._driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO {uuid: $dup}]->()
            SET e.invalid_at = coalesce(e.invalid_at, datetime()),
                e.expired_at = datetime(),
                e.merged_into = $canonical,
                e.curation_reason = 'merged duplicate'
            RETURN e.uuid AS uuid
            """,
            dup=duplicate_uuid, canonical=canonical_uuid,
        )
        ok = bool(res.records)
        try:
            await self.backup.verify_no_loss(snapshot)
        except CurationSafetyError:
            logger.error(
                "merge_duplicate: zero-loss invariant violated after merging %s into %s — "
                "see backup at %s", duplicate_uuid, canonical_uuid, backup_path,
            )
            raise
        return ApplyResult(
            ok=ok, action="merge", edge_uuid=duplicate_uuid, backup_path=str(backup_path),
            detail=f"superseded; merged into {canonical_uuid}" if ok else "duplicate edge not found",
        )

    async def archive(self, edge_uuid: str) -> ApplyResult:
        """Reversibly hide a fact (NOT delete, NOT a temporal end)."""
        # Scoped one-hop backup around the archived edge (WP-H item 2).
        snapshot = await self.backup.collect(edge_uuids=[edge_uuid])
        backup_path = await self.backup.snapshot("archive", edge_uuids=[edge_uuid])
        res = await self._driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO {uuid: $id}]->()
            SET e.archived = true, e.archived_at = datetime()
            RETURN e.uuid AS uuid
            """,
            id=edge_uuid,
        )
        ok = bool(res.records)
        try:
            await self.backup.verify_no_loss(snapshot)
        except CurationSafetyError:
            logger.error(
                "archive: zero-loss invariant violated after archiving %s — see backup at %s",
                edge_uuid, backup_path,
            )
            raise
        return ApplyResult(ok=ok, action="archive", edge_uuid=edge_uuid, backup_path=str(backup_path),
                           detail="archived (reversible)" if ok else "edge not found")

    async def restore(self, edge_uuid: str) -> ApplyResult:
        """Undo an archive (and clear a merge marker): re-activate the fact."""
        res = await self._driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO {uuid: $id}]->()
            SET e.archived = false, e.merged_into = null, e.curation_reason = null
            REMOVE e.archived_at
            RETURN e.uuid AS uuid
            """,
            id=edge_uuid,
        )
        ok = bool(res.records)
        return ApplyResult(ok=ok, action="restore", edge_uuid=edge_uuid,
                           detail="restored" if ok else "edge not found")


def build_curation_engine(graphiti, backup_dir: str = "backups") -> CurationEngine:
    return CurationEngine(graphiti, BackupService(graphiti, backup_dir))
