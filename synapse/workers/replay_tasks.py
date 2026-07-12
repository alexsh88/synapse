"""Celery WRITE task — replay PendingCapture nodes queued during Ollama outages.

Separated from curation_tasks.py because curation tasks are STRICTLY read-only
(the safety contract in celery_app.py / docs/architecture/curation.md). This
module owns the one mutating scheduled task: retrying queued writes.

Beat schedule: every 10 minutes (registered in celery_app.py).
"""

from __future__ import annotations

import asyncio
import logging

from synapse.workers.celery_app import celery_app

logger = logging.getLogger("synapse.replay_tasks")

# Maximum per-node retry attempts before giving up.
_MAX_RETRIES = 5


@celery_app.task(name="replay_pending_captures")
def replay_pending_captures() -> dict:
    """Scan Neo4j for PendingCapture nodes with status='pending_replay' and retry them."""
    return asyncio.run(_replay_pending_captures())


async def _replay_pending_captures() -> dict:
    from synapse.core.knowledge_engine import build_graphiti
    from synapse.core.write_pipeline import build_write_pipeline

    graphiti = build_graphiti()
    try:
        pipeline = build_write_pipeline(graphiti)
        driver = graphiti.driver
        return await _run_replay(driver, pipeline)
    finally:
        await graphiti.close()


async def _run_replay(driver, pipeline) -> dict:
    """Core replay logic; driver + pipeline injected so tests can call this directly."""
    # Fetch pending nodes (up to 100 per run to avoid long lock times).
    result = await driver.execute_query(
        """
        MATCH (p:PendingCapture {status: 'pending_replay'})
        RETURN p.uuid AS uuid, p.hash AS hash, p.content AS content,
               p.type AS type, p.project_id AS project_id,
               p.retry_count AS retry_count
        ORDER BY p.created_at ASC LIMIT 100
        """,
    )
    records = result.records
    if not records:
        logger.info("replay_pending_captures: nothing to replay")
        return {"replayed": 0, "failed": 0, "gave_up": 0}

    replayed = failed = gave_up = 0

    for rec in records:
        node_uuid: str = rec["uuid"]
        content: str = rec["content"] or ""
        ktype: str = rec["type"] or "entity"
        project_id: str | None = rec["project_id"]
        retry_count: int = int(rec["retry_count"] or 0)
        node_hash: str = rec["hash"] or ""

        # Derive scope: project_X → strip prefix; no project → global.
        pid: str | None = None
        if project_id and project_id.startswith("project_"):
            pid = project_id[len("project_"):]
        elif project_id and project_id != "global":
            pid = project_id

        try:
            wr = await pipeline.remember(
                content,
                knowledge_type=ktype,
                project_id=pid,
                force=True,  # skip the triage filter; already triaged on first attempt
            )
            if wr.degraded and wr.reason and "queued: embedder unavailable" in wr.reason:
                # Embedder still down — update retry_count / give-up status.
                raise RuntimeError("embedder still unavailable")

            # Success — mark replayed.
            await driver.execute_query(
                "MATCH (p:PendingCapture {uuid: $uuid}) SET p.status = 'replayed'",
                uuid=node_uuid,
            )
            logger.info("replay success: uuid=%s hash=%.12s", node_uuid, node_hash)
            replayed += 1

        except Exception as exc:  # noqa: BLE001
            new_retry = retry_count + 1
            if new_retry >= _MAX_RETRIES:
                await driver.execute_query(
                    """
                    MATCH (p:PendingCapture {uuid: $uuid})
                    SET p.status = 'replay_failed', p.retry_count = $rc
                    """,
                    uuid=node_uuid,
                    rc=new_retry,
                )
                logger.warning(
                    "replay gave up after %d attempts: uuid=%s (%s)",
                    new_retry, node_uuid, str(exc)[:120],
                )
                gave_up += 1
            else:
                await driver.execute_query(
                    "MATCH (p:PendingCapture {uuid: $uuid}) SET p.retry_count = $rc",
                    uuid=node_uuid,
                    rc=new_retry,
                )
                logger.info(
                    "replay attempt %d/%d failed: uuid=%s (%s)",
                    new_retry, _MAX_RETRIES, node_uuid, str(exc)[:120],
                )
                failed += 1

    summary = {"replayed": replayed, "failed": failed, "gave_up": gave_up}
    logger.info("replay_pending_captures: %s", summary)
    return summary
