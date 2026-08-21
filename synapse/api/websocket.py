"""WebSocket endpoint — real-time knowledge events (Phase 5)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from synapse.api.deps import require_ws_api_key
from synapse.api.events import bus

logger = logging.getLogger("synapse.api.ws")

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    # Gated like every REST router. Without this the one auth mechanism the project has
    # protected the queries while this socket streamed the answers to anyone on the port.
    if not await require_ws_api_key(websocket):
        return
    await websocket.accept()
    await websocket.send_json({"type": "hello"})
    queue = bus.subscribe()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event.model_dump())
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("websocket error")
    finally:
        bus.unsubscribe(queue)
