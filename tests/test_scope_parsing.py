"""One scope parser, shared by every interface.

Three divergent copies had grown — the MCP tools understood clusters, the knowledge route only
stripped a `project_` prefix, and the search route prepended `project_` to whatever it was handed.
Two of them silently mangled a cluster scope: `cluster:trading` became a PROJECT id, so writes died
on the group_id `project_cluster:trading` and searches quietly matched nothing (2026-07-27).
"""

from __future__ import annotations

import pytest

from synapse.api.routes import knowledge as knowledge_route
from synapse.core.schema import GLOBAL_SCOPE, Scope
from synapse.mcp import tools as mcp_tools


# --- the parser -----------------------------------------------------------------------

@pytest.mark.parametrize(
    ("scope", "cluster", "project"),
    [
        (None, None, None),                      # HTTP has no ambient project
        ("", None, None),
        ("   ", None, None),
        ("global", None, None),
        ("GLOBAL", None, None),
        ("cluster:trading", "trading", None),
        ("cluster_trading", "trading", None),
        ("CLUSTER:Trading", "trading", None),    # cluster names normalize to lowercase
        ("project:acme-api", None, "acme-api"),
        ("project_acme-api", None, "acme-api"),
        ("acme-api", None, "acme-api"),
        ("  acme-api  ", None, "acme-api"),
        ("acme-etl", None, "acme-etl"),
        # A project whose name merely STARTS with a cluster-ish word is not a cluster.
        ("clustered-cache", None, "clustered-cache"),
    ],
)
def test_parse_request(scope, cluster, project):
    assert Scope.parse_request(scope) == (cluster, project)


def test_cluster_and_project_are_mutually_exclusive():
    """A cluster write is domain-scoped, so it must never also carry a project."""
    for scope in ("cluster:trading", "cluster_creative"):
        cluster, project = Scope.parse_request(scope)
        assert cluster and project is None


def test_project_case_is_preserved_but_the_prefix_is_not():
    """Project ids are identifiers on disk; only the prefix match is case-insensitive."""
    assert Scope.parse_request("PROJECT_MindTales") == (None, "MindTales")


def test_default_project_applies_only_to_an_omitted_scope():
    """The asymmetry that makes `default_project` a parameter: MCP has a seat, HTTP does not."""
    assert Scope.parse_request(None, default_project="acme-api") == (None, "acme-api")
    # An explicit scope always wins over the default, including explicit global.
    assert Scope.parse_request("global", default_project="acme-api") == (None, None)
    assert Scope.parse_request("acme-sim", default_project="acme-api") == (None, "acme-sim")
    assert Scope.parse_request("cluster:trading", default_project="acme-api") == ("trading", None)


@pytest.mark.parametrize(
    ("scope", "group_id"),
    [
        (None, None),                            # unrestricted — search every scope
        ("", None),
        ("global", GLOBAL_SCOPE),
        ("cluster_trading", "cluster_trading"),
        ("cluster:trading", "cluster_trading"),  # THE BUG: was `project_cluster:trading`
        ("project_acme-api", "project_acme-api"),
        ("acme-api", "project_acme-api"),
    ],
)
def test_group_id_for_request(scope, group_id):
    assert Scope.group_id_for_request(scope) == group_id


def test_every_group_id_is_a_legal_graphiti_group_id():
    """Graphiti allows only [A-Za-z0-9_-]; a colon reaching a group_id is what broke the write."""
    import re

    for scope in ("global", "cluster:trading", "cluster_trading", "project:acme-api", "acme-api"):
        group_id = Scope.group_id_for_request(scope)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", group_id), f"{scope} -> {group_id}"


# --- the interfaces agree -------------------------------------------------------------

@pytest.mark.parametrize(
    "scope", ["global", "cluster:trading", "cluster_trading", "project_acme-api", "acme-api"],
)
def test_mcp_helpers_delegate_to_the_shared_parser(scope):
    assert mcp_tools._scope_to_cluster(scope) == Scope.parse_request(scope)[0]
    assert mcp_tools._scope_to_project(scope, None) == Scope.parse_request(scope)[1]


def test_a_cluster_scope_never_becomes_a_project_id():
    """The regression pin. Returning the literal here is what produced `project_cluster:trading`."""
    assert mcp_tools._scope_to_project("cluster:trading", "acme-api") is None
    assert mcp_tools._scope_to_project("cluster_trading", "acme-api") is None


def test_the_write_route_no_longer_has_its_own_parser():
    """It had one that only knew `project_`; a second copy is how the interfaces drifted."""
    assert not hasattr(knowledge_route, "_project_from_scope")


# --- the write path receives the cluster ----------------------------------------------

class _Result:
    outcome = type("_O", (), {"value": "stored"})()
    scope = "cluster_trading"
    episode_uuid = "ep-1"
    facts: list[str] = []


class _Engine:
    def __init__(self):
        self.calls: list[dict] = []

    async def remember(self, content, **kwargs):
        self.calls.append({"content": content, **kwargs})
        return _Result()

    async def recall(self, query, **kwargs):
        self.calls.append({"recall": query, **kwargs})
        return []

    async def search(self, query, **kwargs):
        self.calls.append({"search": query, **kwargs})
        return []


async def test_the_write_route_forwards_a_cluster_scope():
    engine = _Engine()
    body = knowledge_route.RememberBody(
        content="broker gateway topology", type="convention", scope="cluster:trading")
    await knowledge_route.remember(body, engine=engine)
    (call,) = engine.calls
    assert call["cluster"] == "trading"
    assert call["project_id"] is None


async def test_the_write_route_still_defaults_to_global():
    engine = _Engine()
    await knowledge_route.remember(
        knowledge_route.RememberBody(content="BigDecimal for money in Java"), engine=engine)
    (call,) = engine.calls
    assert call["cluster"] is None and call["project_id"] is None


async def test_the_write_route_still_forwards_a_project_scope():
    engine = _Engine()
    await knowledge_route.remember(
        knowledge_route.RememberBody(content="x", scope="acme-api"), engine=engine)
    (call,) = engine.calls
    assert call["project_id"] == "acme-api" and call["cluster"] is None


# --- recall must not silently answer a different question -----------------------------

async def test_mcp_recall_with_a_cluster_scope_targets_that_cluster():
    """`recall` composes from a PROJECT seat, so it cannot hold an explicit cluster.

    Neither old behaviour was acceptable: the bogus `project_cluster:trading` group_id matched
    nothing silently, and falling through with project_id=None would recall from GLOBAL while
    dropping the requested cluster. It routes to the one path that can target a single tier.
    """
    engine = _Engine()
    await mcp_tools.recall(engine, "acme-api", "gateway ports", scope="cluster:trading")
    (call,) = engine.calls
    assert "search" in call, "a cluster recall must not go to the project-seat recall path"
    assert call["group_ids"] == ["cluster_trading"]


async def test_mcp_recall_without_a_cluster_is_unchanged():
    engine = _Engine()
    await mcp_tools.recall(engine, "acme-api", "gateway ports")
    (call,) = engine.calls
    assert "recall" in call
    assert call["project_id"] == "acme-api"
    assert call["feedback"] is True
