"""Obsidian export — a READ-ONLY mirror of the graph as a markdown vault (Phase 12).

Neo4j stays the source of truth (R3). This renders the *active* knowledge graph to a folder
of markdown notes — one per entity node — with YAML frontmatter (scope/type/temporal, Bases-
friendly) and `[[wikilinks]]` for edges, so you get Obsidian's graph view, search, and Bases
"for free", plus offline/mobile reading. One-directional: nothing is ever read back (avoids the
concurrent-write hazard R3 warns about). Doubles as a human-readable backup.

Design + rationale: docs/research/obsidian-integration.md (option A).

Regeneration: the export OWNS its output folder — it wipes the managed vault dir and rewrites it,
so deleted/superseded knowledge doesn't linger. Point it at a *generated* dir, never a hand-kept vault.
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_KNOWLEDGE = ("Decision", "Convention", "Lesson", "Research", "Pattern", "Tool", "Project")
_ILLEGAL = re.compile(r'[<>:"/\\|?#^\[\]*\x00-\x1f]+')  # filesystem- + Obsidian-illegal chars


def _type_of(labels) -> str:
    for label in labels or []:
        if label != "Entity":
            return label.lower()
    return "entity"


def _native(v):
    return v.to_native() if hasattr(v, "to_native") else v


def _slug(name: str) -> str:
    s = _ILLEGAL.sub(" ", name or "untitled").strip()
    s = re.sub(r"\s+", " ", s)
    return (s[:80].rstrip() or "untitled")


def _pretty_scope(scope: str) -> str:
    return "global" if scope == "global" else scope.replace("project_", "").replace("agent_", "agent-")


class ObsidianExporter:
    def __init__(self, graphiti, out_dir: Path | str = "exports/obsidian") -> None:
        self._driver = graphiti.driver
        self.out = Path(out_dir)

    async def export(self) -> dict:
        nodes = await self._fetch_nodes()
        edges = await self._fetch_edges()

        # Unique filename stem per node (Obsidian wikilinks resolve by basename vault-wide).
        stems: dict[str, str] = {}
        seen: dict[str, int] = defaultdict(int)
        for n in nodes:
            base = _slug(n["name"])
            key = base.lower()
            seen[key] += 1
            stems[n["uuid"]] = base if seen[key] == 1 else f"{base} ({n['uuid'][:6]})"

        out_edges: dict[str, list] = defaultdict(list)
        in_edges: dict[str, list] = defaultdict(list)
        for e in edges:
            if e["source"] in stems and e["target"] in stems:
                out_edges[e["source"]].append(e)
                in_edges[e["target"]].append(e)

        # The exporter owns its dir — wipe + rewrite so stale notes never linger.
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True, exist_ok=True)

        per_scope: dict[str, int] = defaultdict(int)
        for n in nodes:
            folder = self.out / _pretty_scope(n["scope"])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{stems[n['uuid']]}.md").write_text(
                self._render(n, stems, out_edges[n["uuid"]], in_edges[n["uuid"]]),
                encoding="utf-8",
            )
            per_scope[_pretty_scope(n["scope"])] += 1

        self._write_index(per_scope, len(nodes), len(edges))
        return {"notes": len(nodes), "edges": len(edges), "scopes": dict(per_scope),
                "out_dir": str(self.out)}

    # --- rendering -----------------------------------------------------------

    def _render(self, n, stems, outs, ins) -> str:
        ntype = _type_of(n["labels"])
        scope = n["scope"]
        created = _native(n["created"])
        created_s = created.date().isoformat() if isinstance(created, datetime) else ""
        # YAML frontmatter (Obsidian Properties / Bases-friendly).
        fm = [
            "---",
            f"title: {self._yaml(n['name'])}",
            f"uuid: {n['uuid']}",
            f"type: {ntype}",
            f"scope: {scope}",
            f"project: {_pretty_scope(scope)}",
            f"degree: {n['degree']}",
        ]
        if created_s:
            fm.append(f"created: {created_s}")
        fm += ["tags:", f"  - {ntype}", f"  - scope/{_pretty_scope(scope)}", "---", ""]

        body = [f"# {n['name']}", ""]
        if n.get("summary"):
            body += [n["summary"].strip(), ""]
        if outs or ins:
            body.append("## Connections")
            for e in outs:
                body.append(f"- **{e['name'] or 'relates to'}** → [[{stems[e['target']]}]]"
                            + (f" — {e['fact']}" if e.get("fact") else ""))
            for e in ins:
                body.append(f"- [[{stems[e['source']]}]] **{e['name'] or 'relates to'}** → this"
                            + (f" — {e['fact']}" if e.get("fact") else ""))
            body.append("")
        return "\n".join(fm + body)

    @staticmethod
    def _yaml(value: str) -> str:
        v = (value or "").replace('"', "'")
        return f'"{v}"' if re.search(r'[:#\[\]{}]', v) else v

    def _write_index(self, per_scope, n_notes, n_edges) -> None:
        lines = [
            "---", "title: Synapse — knowledge mirror", "---", "",
            "# Synapse — knowledge mirror", "",
            "> **Read-only, generated** from the Neo4j graph (the source of truth). Do not edit by "
            "hand — changes are overwritten on the next export. Open this folder as an Obsidian vault.",
            "",
            f"- Notes: **{n_notes}**  ·  active fact-links: **{n_edges}**",
            "- Each note = one knowledge node; `[[links]]` = graph edges; folders = scope.",
            "- Try the **graph view** (the whole brain) and **Bases** over the `type` / `scope` properties.",
            "", "## Scopes",
        ]
        for scope, c in sorted(per_scope.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{scope}` — {c} notes")
        (self.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- queries -------------------------------------------------------------

    async def _fetch_nodes(self) -> list[dict]:
        res = await self._driver.execute_query(
            """
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r:RELATES_TO]-() WHERE r.invalid_at IS NULL
            WITH n, count(r) AS degree
            RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
                   n.group_id AS scope, n.summary AS summary, n.created_at AS created, degree
            ORDER BY degree DESC
            """,
        )
        return [
            {"uuid": r["uuid"], "name": r["name"], "labels": r["labels"], "scope": r["scope"],
             "summary": r["summary"], "created": r["created"], "degree": int(r["degree"])}
            for r in res.records if r["uuid"] and r["name"]
        ]

    async def _fetch_edges(self) -> list[dict]:
        res = await self._driver.execute_query(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE r.invalid_at IS NULL AND coalesce(r.archived, false) = false
            RETURN a.uuid AS source, b.uuid AS target, r.name AS name, r.fact AS fact
            """,
        )
        return [{"source": r["source"], "target": r["target"], "name": r["name"], "fact": r["fact"]}
                for r in res.records]
