"""Delete Synapse smoke/test knowledge from Neo4j — scoped to named groups only.

Safe by construction: only deletes nodes whose ``group_id`` is in the target set
(plus the legacy ``:SynapseSmoke`` probe node). Never touches real knowledge.

    python -m scripts.clear_test_data                 # clears the default smoke groups
    python -m scripts.clear_test_data project_foo bar  # clears the given groups
"""

from __future__ import annotations

import asyncio
import sys

from graphiti_core.driver.neo4j_driver import Neo4jDriver

from synapse.config import settings

DEFAULT_GROUPS = [
    "project_writetest",
    "project_synapse-demo",
    "project_retrievetest",
    "synapse-smoke",
    "synapse-test",
]


async def main(groups: list[str]) -> int:
    driver = Neo4jDriver(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database
    )
    try:
        before = await driver.execute_query(
            "MATCH (n) WHERE n.group_id IN $g RETURN count(n) AS c", g=groups
        )
        count = before.records[0]["c"]
        await driver.execute_query("MATCH (n) WHERE n.group_id IN $g DETACH DELETE n", g=groups)
        # legacy probe node from hello_knowledge.py (no group_id)
        await driver.execute_query("MATCH (n:SynapseSmoke) DETACH DELETE n")
        print(f"[clear] deleted {count} node(s) across groups: {groups}")
        print("[clear] also removed any :SynapseSmoke probe node")
        return 0
    finally:
        await driver.close()


if __name__ == "__main__":
    groups = sys.argv[1:] or DEFAULT_GROUPS
    raise SystemExit(asyncio.run(main(groups)))
