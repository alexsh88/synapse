"""Measure fact-to-fact dedup precision across thresholds (R7 tuning, Phase 10/11).

One similarity scan (pairs >= 0.90), then cluster counts at several thresholds plus a
sample of pairs in the risky [0.90, 0.94) band so the threshold can be chosen from data,
not guessed. Read-only. Cosine is computed in Neo4j over stored embeddings (no Ollama).

    python -m scripts.analyze_dedup
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from synapse.core.curation_engine import build_curation_engine
from synapse.core.knowledge_engine import build_graphiti

THRESHOLDS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]


def _clusters(pairs, th):
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    members = set()
    for a, b, s, *_ in pairs:
        if s >= th:
            parent[find(a)] = find(b)
            members.update((a, b))
    comps = defaultdict(list)
    for m in members:
        comps[find(m)].append(m)
    return comps, members


async def main() -> int:
    graphiti = build_graphiti()
    try:
        eng = build_curation_engine(graphiti)
        # Direct, UNBOUNDED query (the engine's _similar_pairs caps at 100 — that's what we're measuring).
        res = await graphiti.driver.execute_query(
            """
            MATCH (a:Entity)-[e1:RELATES_TO]->(b:Entity)
            MATCH (c:Entity)-[e2:RELATES_TO]->(d:Entity)
            WHERE e1.group_id = e2.group_id AND e1.uuid < e2.uuid
              AND e1.invalid_at IS NULL AND e2.invalid_at IS NULL
              AND coalesce(e1.archived, false) = false AND coalesce(e2.archived, false) = false
              AND e1.fact_embedding IS NOT NULL AND e2.fact_embedding IS NOT NULL
            WITH e1, e2, vector.similarity.cosine(e1.fact_embedding, e2.fact_embedding) AS sim
            WHERE sim >= 0.90
            RETURN e1.uuid AS a_uuid, e1.fact AS a_fact, e2.uuid AS b_uuid, e2.fact AS b_fact,
                   e1.group_id AS scope, sim
            ORDER BY sim DESC
            """,
        )
        rows = res.records
        pairs = [(r["a_uuid"], r["b_uuid"], float(r["sim"]), r["a_fact"], r["b_fact"], r["scope"])
                 for r in rows]
        print(f"pairs with cosine >= 0.90 (UNBOUNDED): {len(pairs)}")
        for th in THRESHOLDS:
            comps, members = _clusters(pairs, th)
            merge_candidates = sum(len(v) - 1 for v in comps.values())
            print(f"  th={th:.2f}: {len(comps):3} clusters, {merge_candidates:3} merge candidates, "
                  f"{len(members):3} facts involved")

        band = sorted((p for p in pairs if 0.90 <= p[2] < 0.95), key=lambda p: p[2])
        print(f"\nrisky band [0.90, 0.95): {len(band)} pairs — eyeball for false positives:")
        for a, b, s, fa, fb, sc in band[:12]:
            print(f"  {s:.3f} [{sc}]")
            print(f"     A: {fa[:96]}")
            print(f"     B: {fb[:96]}")
        return 0
    finally:
        await graphiti.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
