"""Liveness vs readiness, and WebSocket auth (2026-08 hardening pass).

Kept out of test_api.py because these exercise ``app.state`` and the handshake rather than a
route behind a dependency override. That distinction is the point: ``client_engine`` there never
sets ``app.state.engine``, which is exactly the state ``/readyz`` has to notice and ``/health``
cannot.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from synapse.api.main import app
from synapse.config import settings


class _Driver:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []

    async def execute_query(self, query, **kw):
        self.queries.append(query)
        if self.fail:
            raise ConnectionError("bolt refused at 10.0.0.5:7687")
        return None


class _Redis:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.pings = 0

    async def ping(self):
        self.pings += 1
        if self.fail:
            raise ConnectionError("redis refused")
        return True


class _Reader:
    def __init__(self, redis) -> None:
        self.redis = redis


class _Graphiti:
    def __init__(self, driver) -> None:
        self.driver = driver


class _Engine:
    """The two attributes /readyz reaches through: graphiti.driver and reader.redis."""

    def __init__(self, *, driver_fails: bool = False, redis=None) -> None:
        self.graphiti = _Graphiti(_Driver(fail=driver_fails))
        self.reader = _Reader(redis)


@pytest.fixture
def client():
    c = TestClient(app)
    yield c
    # app.state is process-wide; a leaked engine would make the next test's "not connected"
    # case pass for the wrong reason.
    if hasattr(app.state, "engine"):
        delattr(app.state, "engine")


# ── liveness / readiness ──────────────────────────────────────────────────────

def test_health_answers_ok_with_no_engine_at_all(client):
    """Liveness is deliberately dependency-free. If this ever starts checking things, the
    container healthcheck and the readiness probe collapse back into one signal."""
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_readyz_is_503_before_connect(client):
    r = client.get("/api/v1/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["engine"] == "not connected"


def test_readyz_is_200_when_every_dependency_answers(client):
    app.state.engine = _Engine(redis=_Redis())
    r = client.get("/api/v1/readyz")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ready"] is True
    assert body["checks"] == {"engine": "ok", "neo4j": "ok", "redis": "ok"}


def test_readyz_names_neo4j_when_bolt_is_down(client):
    """The case /health reported as ok: process up, graph unreachable."""
    app.state.engine = _Engine(driver_fails=True, redis=_Redis())
    r = client.get("/api/v1/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["neo4j"].startswith("error: ConnectionError")


def test_readyz_fails_when_a_configured_redis_is_unreachable(client):
    app.state.engine = _Engine(redis=_Redis(fail=True))
    r = client.get("/api/v1/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["redis"].startswith("error:")


def test_unconfigured_redis_is_reported_without_blocking_readiness(client):
    """Redis only caches briefs, so running without it is a configuration, not an outage."""
    app.state.engine = _Engine(redis=None)
    r = client.get("/api/v1/readyz")
    assert r.status_code == 200, r.text
    assert r.json()["checks"]["redis"] == "not configured"


def test_readyz_reports_the_exception_type_not_its_message(client):
    """Probes get exposed further than the API they front — an address in the body is a leak."""
    app.state.engine = _Engine(driver_fails=True, redis=_Redis())
    assert "10.0.0.5" not in client.get("/api/v1/readyz").text


def test_readyz_actually_queries_neo4j_rather_than_inspecting_state(client):
    """Guards the regression that started this: a check that never touches the dependency."""
    engine = _Engine(redis=_Redis())
    app.state.engine = engine
    client.get("/api/v1/readyz")
    assert engine.graphiti.driver.queries == ["RETURN 1"]
    assert engine.reader.redis.pings == 1


# ── WebSocket auth ────────────────────────────────────────────────────────────

def test_ws_is_open_when_no_api_key_is_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"


def test_ws_is_rejected_when_a_key_is_configured_and_absent(client, monkeypatch):
    """The gap this closes: /ws broadcast every write while the REST API demanded a key."""
    monkeypatch.setattr(settings, "api_key", "secret123")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()


def test_ws_is_rejected_when_the_key_is_wrong(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws?key=nope") as ws:
            ws.receive_json()


def test_ws_accepts_the_key_from_a_query_param(client, monkeypatch):
    """Browsers cannot set headers on a WebSocket handshake, so header-only auth here would
    mean "authenticated for scripts, unreachable for the UI"."""
    monkeypatch.setattr(settings, "api_key", "secret123")
    with client.websocket_connect("/ws?key=secret123") as ws:
        assert ws.receive_json()["type"] == "hello"


def test_ws_accepts_the_key_from_a_header(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")
    with client.websocket_connect("/ws", headers={"X-Synapse-Key": "secret123"}) as ws:
        assert ws.receive_json()["type"] == "hello"
