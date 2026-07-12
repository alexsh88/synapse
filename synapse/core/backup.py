"""Backup + zero-loss verification for curation (R8).

Every mutating curation operation snapshots the affected graph first, and a
``verify_no_loss`` check proves that no fact (or entity) present before an
operation has *vanished* afterward. Curation never hard-deletes — facts are
superseded (``invalid_at``) or flagged (``archived``) — so this check should
always pass; its job is to *catch a regression* that ever introduced a delete.

See ``docs/architecture/curation.md`` for the full safety contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class CurationSafetyError(RuntimeError):
    """Raised when a zero-loss invariant is violated — knowledge would be lost."""


class GraphSnapshot(BaseModel):
    taken_at: str
    edges: list[dict] = Field(default_factory=list)
    node_uuids: list[str] = Field(default_factory=list)

    @property
    def edge_uuids(self) -> list[str]:
        return [e["uuid"] for e in self.edges]


class BackupService:
    """Read-only graph export + a before/after no-loss diff."""

    def __init__(self, graphiti, backup_dir: Path | str = "backups") -> None:
        self._driver = graphiti.driver
        self._dir = Path(backup_dir)

    async def collect(self, edge_uuids: list[str] | None = None) -> GraphSnapshot:
        """Snapshot the graph.

        Full-graph (``edge_uuids=None``) exports every RELATES_TO edge + every Entity node — unchanged.

        Scoped (``edge_uuids`` given) exports only the ONE-HOP NEIGHBORHOOD of those edges: the named
        edges, their endpoint nodes, and every edge incident to those endpoints. That's the exact region a
        merge/archive can touch, so ``verify_no_loss`` still catches any loss there while the snapshot cost
        is O(neighborhood) not O(corpus) (WP-H item 2). ``verify_no_loss`` stays correct because it only
        checks that what IS in the snapshot still exists — a subset snapshot verifies a subset, never
        producing a false "loss" for an untouched edge it never recorded.
        """
        if edge_uuids is not None:
            edge_res = await self._driver.execute_query(
                """
                MATCH (a:Entity)-[e0:RELATES_TO]->(b:Entity)
                WHERE e0.uuid IN $edge_uuids
                WITH collect(DISTINCT a) + collect(DISTINCT b) AS endpoints
                UNWIND endpoints AS ep
                MATCH (ep)-[e:RELATES_TO]-(:Entity)
                RETURN DISTINCT e.uuid AS uuid, e.fact AS fact, e.group_id AS group_id,
                       toString(e.valid_at) AS valid_at, toString(e.invalid_at) AS invalid_at,
                       e.archived AS archived, e.merged_into AS merged_into
                """,
                edge_uuids=edge_uuids,
            )
            node_res = await self._driver.execute_query(
                """
                MATCH (a:Entity)-[e0:RELATES_TO]->(b:Entity)
                WHERE e0.uuid IN $edge_uuids
                WITH collect(DISTINCT a) + collect(DISTINCT b) AS endpoints
                UNWIND endpoints AS ep
                RETURN DISTINCT ep.uuid AS uuid
                """,
                edge_uuids=edge_uuids,
            )
        else:
            edge_res = await self._driver.execute_query(
                """
                MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
                RETURN e.uuid AS uuid, e.fact AS fact, e.group_id AS group_id,
                       toString(e.valid_at) AS valid_at, toString(e.invalid_at) AS invalid_at,
                       e.archived AS archived, e.merged_into AS merged_into
                """,
            )
            node_res = await self._driver.execute_query(
                "MATCH (n:Entity) RETURN n.uuid AS uuid",
            )
        edges = [
            {
                "uuid": r["uuid"], "fact": r["fact"], "group_id": r["group_id"],
                "valid_at": r["valid_at"], "invalid_at": r["invalid_at"],
                "archived": r["archived"], "merged_into": r["merged_into"],
            }
            for r in edge_res.records
        ]
        return GraphSnapshot(
            taken_at=datetime.now(timezone.utc).isoformat(),
            edges=edges,
            node_uuids=[r["uuid"] for r in node_res.records],
        )

    async def snapshot(self, label: str = "curation", edge_uuids: list[str] | None = None) -> Path:
        """Write a timestamped JSON snapshot and return its path.

        Pass ``edge_uuids`` for a scoped one-hop snapshot (see :meth:`collect`); omit for a full-graph one.
        """
        snap = await self.collect(edge_uuids)
        self._dir.mkdir(parents=True, exist_ok=True)
        stamp = snap.taken_at.replace(":", "").replace("-", "").replace(".", "_")
        path = self._dir / f"{label}-{stamp}.json"
        path.write_text(snap.model_dump_json(indent=2), encoding="utf-8")
        return path

    async def verify_no_loss(self, before: GraphSnapshot | Path | str) -> dict:
        """Assert every edge/node uuid in ``before`` still resolves. Raise if not."""
        if isinstance(before, (str, Path)):
            before = GraphSnapshot.model_validate_json(Path(before).read_text(encoding="utf-8"))
        current = await self.collect()
        missing_edges = sorted(set(before.edge_uuids) - set(current.edge_uuids))
        missing_nodes = sorted(set(before.node_uuids) - set(current.node_uuids))
        if missing_edges or missing_nodes:
            raise CurationSafetyError(
                f"zero-loss violated: {len(missing_edges)} fact(s) and "
                f"{len(missing_nodes)} entity node(s) vanished since backup "
                f"(taken {before.taken_at}). First missing fact: "
                f"{missing_edges[0] if missing_edges else missing_nodes[0]}"
            )
        return {
            "ok": True,
            "edges_checked": len(before.edge_uuids),
            "nodes_checked": len(before.node_uuids),
        }
