"""CurationEngine — analysis + reversible, backup-first mutations (R8/R4).

Fake driver, no live Neo4j. The safety guarantees under test:
- mutations snapshot a backup BEFORE issuing any write;
- mutations are SET/temporal, never DELETE (verified by inspecting the issued query);
- archive → restore round-trips.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from synapse.core.backup import BackupService
from synapse.core.curation_engine import CurationEngine

UTC = timezone.utc


class FakeResult:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    def __init__(self, *, knn_raises: bool = False, **canned):
        self.canned = canned
        self.knn_raises = knn_raises          # simulate the vector-index path being unavailable
        self.knn_calls = 0                    # how many times the k-NN candidate query was issued
        self.scan_calls = 0                   # how many times the brute-force fallback scan was issued
        self.writes: list[tuple[str, dict]] = []

    async def execute_query(self, query, **params):
        q = query
        # --- mutations (checked first; they also mention merged_into/archived) ---
        if "e.merged_into = $canonical" in q:
            self.writes.append(("merge", params, q))
            return FakeResult([{"uuid": params.get("dup")}])
        if "e.archived = true" in q:
            self.writes.append(("archive", params, q))
            return FakeResult([{"uuid": params.get("id")}])
        if "REMOVE e.archived_at" in q:
            self.writes.append(("restore", params, q))
            return FakeResult([{"uuid": params.get("id")}])
        # --- backup collect (scoped one-hop OR full graph) ---
        if "merged_into AS merged_into" in q:
            return FakeResult(self.canned.get("edges", []))
        if "n.uuid AS uuid" in q or "ep.uuid AS uuid" in q:
            return FakeResult(self.canned.get("nodes", []))
        # --- analysis: k-NN candidate path (banded pairs, review band via $ceil) ---
        if "db.index.vector.queryRelationships" in q:
            self.knn_calls += 1
            if self.knn_raises:
                raise RuntimeError("vector index unavailable")
            key = "review" if "sim < $ceil" in q else "pairs"
            return FakeResult(self.canned.get(key, []))
        # --- analysis: brute-force fallback scan (cross-join) ---
        if "e1.uuid < e2.uuid" in q:
            self.scan_calls += 1
            key = "review" if "sim < $ceil" in q else "pairs"
            return FakeResult(self.canned.get(key, []))
        if "$cutoff" in q:
            return FakeResult(self.canned.get("stale", []))
        return FakeResult([])


class _FG:
    def __init__(self, driver):
        self.driver = driver


def engine(tmp_path, **canned):
    drv = FakeDriver(**canned)
    fg = _FG(drv)
    eng = CurationEngine(fg, BackupService(fg, str(tmp_path)),
                         now=datetime(2026, 6, 3, tzinfo=UTC))
    return eng, drv


def _pair(a, b, sim, a_created, b_created, scope="project_acme-api", edges_scanned=2):
    # _edges_scanned is returned by the k-NN candidate query (coverage stat, logged); the brute-force
    # fallback path never reads it, so its presence is harmless there.
    return {"a_uuid": a, "a_fact": f"fact {a}", "a_created": a_created,
            "b_uuid": b, "b_fact": f"fact {b}", "b_created": b_created,
            "scope": scope, "sim": sim, "_edges_scanned": edges_scanned}


def test_curation_thresholds_default_high_and_overridable(tmp_path):
    fg = _FG(FakeDriver())
    e = CurationEngine(fg, BackupService(fg, str(tmp_path)))
    # fact<->fact dedup is stricter than the write-time 0.90 (measured: 0.90 over-merges).
    assert e.dedup_threshold == 0.97 and e.review_floor == 0.90 and e.pair_limit == 500
    e2 = CurationEngine(fg, BackupService(fg, str(tmp_path)),
                        dedup_threshold=0.95, review_floor=0.85, pair_limit=10)
    assert (e2.dedup_threshold, e2.review_floor, e2.pair_limit) == (0.95, 0.85, 10)


async def test_find_duplicates_clusters_chain_and_picks_earliest_canonical(tmp_path):
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    t1 = datetime(2026, 5, 2, tzinfo=UTC)
    t2 = datetime(2026, 5, 3, tzinfo=UTC)
    eng, _ = engine(tmp_path, pairs=[
        _pair("e1", "e2", 0.95, t0, t1),
        _pair("e2", "e3", 0.92, t1, t2),   # chains into one cluster e1~e2~e3
    ])
    clusters = await eng.find_duplicates()
    assert len(clusters) == 1
    c = clusters[0]
    assert c.canonical.uuid == "e1"                      # earliest created
    assert {d.uuid for d in c.duplicates} == {"e2", "e3"}
    assert c.max_similarity == 0.95                      # true cluster max, not 0.92


async def test_find_stale_computes_age(tmp_path):
    old = datetime(2025, 11, 1, tzinfo=UTC)              # ~214 days before injected now
    eng, _ = engine(tmp_path, stale=[
        {"uuid": "e1", "fact": "ancient lesson", "scope": "global", "created_at": old},
    ])
    items = await eng.find_stale(older_than_days=180)
    assert items[0].uuid == "e1" and items[0].age_days == 214


async def test_merge_backs_up_first_and_is_non_destructive(tmp_path):
    eng, drv = engine(tmp_path, edges=[{"uuid": "e1", "fact": "f", "group_id": "global",
                                        "valid_at": None, "invalid_at": None,
                                        "archived": None, "merged_into": None}],
                      nodes=[{"uuid": "n1"}])
    res = await eng.merge_duplicate("e1", "e2")

    assert res.ok and res.action == "merge"
    assert res.backup_path and Path(res.backup_path).exists()     # backup taken first (R8)
    merge_write = next(w for w in drv.writes if w[0] == "merge")
    assert "DELETE" not in merge_write[2].upper()                 # temporal end, never delete (R4)
    assert "e.invalid_at" in merge_write[2]


async def test_archive_then_restore_round_trips(tmp_path):
    eng, drv = engine(tmp_path, edges=[], nodes=[])
    arch = await eng.archive("e9")
    assert arch.ok and Path(arch.backup_path).exists()
    rest = await eng.restore("e9")
    assert rest.ok and rest.action == "restore"
    kinds = [w[0] for w in drv.writes]
    assert kinds == ["archive", "restore"]
    assert "DELETE" not in drv.writes[0][2].upper()


async def test_find_duplicates_uses_knn_index_not_scan(tmp_path):
    # The candidate query goes through the native relationship vector index (WP-H item 1), NOT the old
    # O(n^2) cross-join scan.
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    eng, drv = engine(tmp_path, pairs=[_pair("e1", "e2", 0.98, t0, t0)])
    clusters = await eng.find_duplicates()
    assert len(clusters) == 1 and clusters[0].canonical.uuid == "e1"
    assert drv.knn_calls == 1 and drv.scan_calls == 0     # index path used, scan untouched


async def test_similar_pairs_falls_back_to_scan_when_index_raises(tmp_path):
    # If the vector-index query raises, candidate generation falls back to the brute-force scan and still
    # returns the deduped banded pairs (WP-H item 1 fallback).
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    drv = FakeDriver(knn_raises=True, pairs=[_pair("e1", "e2", 0.98, t0, t0)])
    fg = _FG(drv)
    eng = CurationEngine(fg, BackupService(fg, str(tmp_path)), now=datetime(2026, 6, 3, tzinfo=UTC))
    clusters = await eng.find_duplicates()
    assert len(clusters) == 1 and clusters[0].canonical.uuid == "e1"
    assert drv.knn_calls == 1 and drv.scan_calls == 1     # tried index, fell through to scan
    assert eng._knn_fallback_logged is True               # logged once


async def test_review_pairs_use_knn_band(tmp_path):
    # The gray-band review query also goes through the k-NN path, carrying the $ceil band.
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    eng, drv = engine(tmp_path, review=[_pair("e3", "e4", 0.92, t0, t0)])
    pairs = await eng.find_review_pairs()
    assert len(pairs) == 1 and pairs[0].similarity == 0.92
    assert drv.knn_calls == 1 and drv.scan_calls == 0


async def test_suggestions_bundles_all_three(tmp_path):
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    eng, _ = engine(
        tmp_path,
        pairs=[_pair("e1", "e2", 0.95, t0, t0)],
        review=[_pair("e3", "e4", 0.80, t0, t0)],
        stale=[{"uuid": "e5", "fact": "old", "scope": "global",
                "created_at": datetime(2025, 1, 1, tzinfo=UTC)}],
    )
    s = await eng.suggestions()
    assert len(s.duplicates) == 1 and len(s.review_pairs) == 1 and len(s.stale) == 1
    assert s.review_pairs[0].similarity == 0.80
    assert s.generated_at is not None
