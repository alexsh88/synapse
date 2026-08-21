"""Direct Neo4j access for maintenance scripts — no Graphiti, no LLM, no embedder.

Most code should go through ``KnowledgeEngine`` / ``build_graphiti()``. Maintenance scripts should
not: they only read and rewrite properties, and ``build_graphiti()`` drags in machinery they never
use — an Ollama-backed embedder, an extraction LLM client, and
``build_indices_and_constraints()``, which fires a batch of concurrent index DDL statements on
every startup.

That batch is a real failure mode, not a theoretical one: running the edge-name migration through
``build_graphiti()`` died with ``IncompleteCommit: Failed to read from defunct connection`` inside
the index-DDL gather, against a Neo4j that was healthy and had been up for hours (the intermittent
defunct-connection issue noted in ``synapse/config.py``). A rename script has no business creating
indexes, so it no longer connects that way.

    async with direct_graph() as graph:
        await graph.driver.execute_query("MATCH (n) RETURN count(n) AS n")
        # `graph` is also accepted anywhere a `.driver`-shaped object is expected,
        # e.g. BackupService(graph, "backups")
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase

from synapse.config import settings


class DirectGraph:
    """Minimal stand-in for a Graphiti instance, exposing only ``.driver``.

    Enough for ``BackupService`` and any Cypher-only maintenance path, which is deliberately all
    it offers — anything needing extraction or embeddings should build a real Graphiti.
    """

    def __init__(self, driver) -> None:
        self.driver = driver

    async def close(self) -> None:
        await self.driver.close()


def build_direct_driver():
    """An async Neo4j driver from settings. Caller owns closing it.

    Note ``neo4j_uri`` deliberately uses 127.0.0.1 rather than localhost — see the comment in
    ``synapse/config.py``: letting the driver resolve localhost to IPv6 ``::1`` causes intermittent
    defunct connections against Docker's IPv4 port-forward.
    """
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.require_neo4j_password()),
    )


@asynccontextmanager
async def direct_graph():
    """Scoped :class:`DirectGraph`, closed on exit even if the body raises."""
    graph = DirectGraph(build_direct_driver())
    try:
        yield graph
    finally:
        await graph.close()
