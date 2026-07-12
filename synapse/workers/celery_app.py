"""Celery app for Synapse background curation (Phase 10).

Broker + result backend = Redis (``settings.redis_url``, db on port 6382). Tasks are
**read-only analysis only** — they surface suggestions and cache health; they NEVER
mutate the graph on a timer (the safety contract: destructive intent is always an
explicit, human-approved API call). See ``docs/architecture/curation.md``.

Run a worker:   celery -A synapse.workers.celery_app worker --loglevel=info
Run the beat:   celery -A synapse.workers.celery_app beat --loglevel=info
"""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from synapse.config import settings

celery_app = Celery(
    "synapse",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["synapse.workers.curation_tasks", "synapse.workers.replay_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "nightly-curation-scan": {
            "task": "synapse.curation.scan_suggestions",
            "schedule": crontab(hour=3, minute=0),
        },
        "nightly-health-scan": {
            "task": "synapse.curation.scan_health",
            "schedule": crontab(hour=3, minute=15),
        },
        # WRITE task: retry queued writes from Ollama outages (every 10 minutes).
        "replay-pending-captures": {
            "task": "replay_pending_captures",
            "schedule": crontab(minute="*/10"),
        },
    },
)

# Register tasks on plain import too (not just on worker start via `include`),
# so the API and tests can reach them. curation_tasks/replay_tasks import the app defined above.
from synapse.workers import curation_tasks  # noqa: E402,F401
from synapse.workers import replay_tasks  # noqa: E402,F401
