"""ObsidianExporter — renders the graph to a markdown vault. Fake driver, no live Neo4j."""

from __future__ import annotations

from datetime import datetime, timezone

from synapse.core.obsidian_export import ObsidianExporter


class FakeResult:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    def __init__(self, nodes, edges):
        self._nodes, self._edges = nodes, edges

    async def execute_query(self, query, **params):
        if "n.created_at AS created, degree" in query:
            return FakeResult(self._nodes)
        if "a.uuid AS source, b.uuid AS target" in query:
            return FakeResult(self._edges)
        return FakeResult([])


class _FG:
    def __init__(self, driver):
        self.driver = driver


def _node(uuid, name, label, scope, summary="s"):
    return {"uuid": uuid, "name": name, "labels": ["Entity", label], "scope": scope,
            "summary": summary, "created": datetime(2026, 6, 1, tzinfo=timezone.utc), "degree": 1}


async def test_export_writes_notes_frontmatter_and_wikilinks(tmp_path):
    nodes = [
        _node("u1", "BigDecimal for money", "Convention", "global"),
        _node("u2", "Acme-API exchange", "Decision", "project_acme-api"),
    ]
    edges = [{"source": "u2", "target": "u1", "name": "AppliesTo", "fact": "Acme-API uses BigDecimal"}]
    stats = await ObsidianExporter(_FG(FakeDriver(nodes, edges)), tmp_path / "vault").export()

    assert stats["notes"] == 2 and stats["edges"] == 1
    # folders by pretty scope
    conv = (tmp_path / "vault" / "global" / "BigDecimal for money.md").read_text(encoding="utf-8")
    dec = (tmp_path / "vault" / "acme-api" / "Acme-API exchange.md").read_text(encoding="utf-8")
    assert "type: convention" in conv and "scope: global" in conv and "title: BigDecimal for money" in conv
    assert "type: decision" in dec and "project: acme-api" in dec
    # the edge becomes a wikilink from the source note to the target note
    assert "[[BigDecimal for money]]" in dec and "Acme-API uses BigDecimal" in dec
    # index written
    assert (tmp_path / "vault" / "README.md").exists()


async def test_export_dedupes_colliding_names(tmp_path):
    nodes = [
        _node("aaaaaa11", "React", "Tool", "project_acme-store"),
        _node("bbbbbb22", "React", "Tool", "project_acme-cms"),
    ]
    await ObsidianExporter(_FG(FakeDriver(nodes, [])), tmp_path / "v").export()
    # second "React" gets a uuid-suffixed filename so wikilinks stay unambiguous
    assert (tmp_path / "v" / "acme-store" / "React.md").exists()
    assert (tmp_path / "v" / "acme-cms" / "React (bbbbbb).md").exists()
