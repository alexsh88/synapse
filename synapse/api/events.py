"""In-process event bus for real-time UI updates (Phase 5).

Routes publish a KnowledgeEvent on every successful write; the WebSocket
endpoint subscribes and forwards to connected clients. Single-process only
(fine for a self-hosted personal tool); swap for Redis pub/sub if ever scaled.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

logger = logging.getLogger("synapse.api.events")

# Hard cap per subscriber. A dead/slow WebSocket client will have its queue
# filled to MAXSIZE; subsequent events are dropped (logged as WARNING) rather
# than blocking all publishers during write bursts.
_QUEUE_MAXSIZE = 100


class KnowledgeEvent(BaseModel):
    type: str                    # knowledge.added | knowledge.updated | knowledge.forgotten
                                 #   | project.connect.progress | project.connect.done
    id: str | None = None
    scope: str | None = None
    summary: str | None = None
    state: str | None = None     # project.connect.done: "done" or "error"
    error: str | None = None     # project.connect.done: failure message when state=="error"
    # project.connect.* progress fields (optional)
    done: int | None = None
    total: int | None = None
    stored: int | None = None


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def publish(self, event: KnowledgeEvent) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "EventBus: subscriber queue full (maxsize=%d); dropping event type=%s",
                    _QUEUE_MAXSIZE, event.type,
                )


bus = EventBus()
