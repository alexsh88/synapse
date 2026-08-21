"""Write provenance (roadmap item 13, research §5.2).

The corpus had four distinct source_description values and no writer identity at all — what the
multi-agent memory literature calls provenance collapse. This is the one roadmap item that is cheap
now and impossible retroactively, so the tests focus on: nothing is ever anonymous, precedence is
predictable, and attribution can never break a write.
"""

from __future__ import annotations

import pytest

from synapse.core.provenance import Provenance, resolve


def test_props_are_prefixed_to_avoid_colliding_with_graphitis_schema():
    # Graphiti owns the Episodic node and adds fields across versions; an unprefixed `model` or
    # `host` on a node type we do not control is asking for a future collision.
    props = Provenance(agent="claude-code", model="claude-opus-5", session_id="s1",
                       host="box").as_props()
    assert props == {
        "prov_agent": "claude-code", "prov_model": "claude-opus-5",
        "prov_session_id": "s1", "prov_host": "box",
    }


def test_empty_and_blank_fields_are_omitted_not_written_as_null():
    # Omitting keeps count(e.prov_session_id) a usable coverage measure.
    assert Provenance().as_props() == {}
    assert Provenance(agent="   ", model="").as_props() == {}
    assert Provenance(agent=" claude-code ").as_props() == {"prov_agent": "claude-code"}


def test_is_empty_reflects_whether_anything_would_be_stamped():
    assert Provenance().is_empty()
    assert not Provenance(agent="x").is_empty()


def test_resolve_prefers_explicit_over_keyword_over_environment(monkeypatch):
    monkeypatch.setenv("SYNAPSE_AGENT", "from-env")
    monkeypatch.setenv("SYNAPSE_SESSION_ID", "env-session")
    # explicit wins over both
    p = resolve(Provenance(agent="explicit"), agent="keyword")
    assert p.agent == "explicit"
    # keyword wins over env
    assert resolve(agent="keyword").agent == "keyword"
    # env is the fallback
    assert resolve().agent == "from-env"
    assert resolve().session_id == "env-session"


def test_resolve_treats_blank_environment_values_as_absent(monkeypatch):
    monkeypatch.setenv("SYNAPSE_AGENT", "   ")
    assert resolve().agent is None


def test_resolve_always_attributes_the_host_so_a_write_is_never_anonymous(monkeypatch):
    monkeypatch.delenv("SYNAPSE_AGENT", raising=False)
    monkeypatch.delenv("SYNAPSE_SESSION_ID", raising=False)
    monkeypatch.delenv("SYNAPSE_MODEL", raising=False)
    monkeypatch.delenv("SYNAPSE_PROV_NO_HOST", raising=False)
    p = resolve()
    assert p.host, "host is the last line of attribution"
    assert not p.is_empty()


@pytest.mark.parametrize("optout", ["1", "true", "YES"])
def test_host_can_be_switched_off(monkeypatch, optout):
    # Hostname is the most identifying field here, so it is the one thing that can be disabled.
    monkeypatch.setenv("SYNAPSE_PROV_NO_HOST", optout)
    assert resolve().host is None


def test_a_hostname_failure_never_breaks_attribution(monkeypatch):
    import synapse.core.provenance as prov

    monkeypatch.delenv("SYNAPSE_PROV_NO_HOST", raising=False)
    monkeypatch.setattr(prov.socket, "gethostname", lambda: (_ for _ in ()).throw(OSError("no dns")))
    assert resolve().host is None  # must not raise


async def test_the_write_path_stamps_provenance_alongside_the_content_hash():
    from synapse.core.write_pipeline import WritePipeline

    class _Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **params):
            self.calls.append((query, params))

            class _R:
                records = []

            return _R()

    class _G:
        def __init__(self, driver):
            self.driver = driver

    driver = _Driver()
    pipeline = WritePipeline(graphiti=_G(driver), embedder=None, index=None, triage=None)
    await pipeline._stamp_episode("ep-1", "abc123",
                                  Provenance(agent="claude-code", session_id="s9"))
    stamp = next(p for q, p in driver.calls if "SET e += $props" in q)
    assert stamp["props"]["content_hash"] == "abc123"
    assert stamp["props"]["prov_agent"] == "claude-code"
    assert stamp["props"]["prov_session_id"] == "s9"


async def test_stamping_without_provenance_still_writes_the_hash():
    from synapse.core.write_pipeline import WritePipeline

    class _Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **params):
            self.calls.append((query, params))

            class _R:
                records = []

            return _R()

    class _G:
        def __init__(self, driver):
            self.driver = driver

    driver = _Driver()
    pipeline = WritePipeline(graphiti=_G(driver), embedder=None, index=None, triage=None)
    await pipeline._stamp_episode("ep-1", "abc123", None)
    stamp = next(p for q, p in driver.calls if "SET e += $props" in q)
    assert stamp["props"] == {"content_hash": "abc123"}


async def test_a_stamp_failure_never_fails_the_write():
    from synapse.core.write_pipeline import WritePipeline

    class _Boom:
        async def execute_query(self, query, **params):
            raise RuntimeError("neo4j down")

    class _G:
        driver = _Boom()

    pipeline = WritePipeline(graphiti=_G(), embedder=None, index=None, triage=None)
    await pipeline._stamp_episode("ep-1", "abc", Provenance(agent="x"))  # must not raise


def test_synapse_host_env_overrides_the_container_id(monkeypatch):
    # Verified live: inside Docker, socket.gethostname() returns the container id
    # ("9c03e1a73872") — ephemeral, and useless for saying which machine wrote the knowledge.
    monkeypatch.delenv("SYNAPSE_PROV_NO_HOST", raising=False)
    monkeypatch.setenv("SYNAPSE_HOST", "workstation")
    assert resolve().host == "workstation"


def test_the_optout_still_wins_over_an_explicit_host_env(monkeypatch):
    monkeypatch.setenv("SYNAPSE_HOST", "workstation")
    monkeypatch.setenv("SYNAPSE_PROV_NO_HOST", "1")
    assert resolve().host is None


def test_an_explicitly_passed_host_beats_the_environment(monkeypatch):
    monkeypatch.setenv("SYNAPSE_HOST", "from-env")
    assert resolve(Provenance(host="from-caller")).host == "from-caller"
