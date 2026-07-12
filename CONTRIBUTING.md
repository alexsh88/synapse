# Contributing to Synapse

## Dev setup

**Python backend**

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest tests/ -v
```

The integration tests (`test_api.py`, `test_mcp_tools.py`, etc.) require the stack running:

```bash
docker compose up -d
```

**React UI**

```bash
cd ui
npm install
npm run dev        # dev server at http://localhost:5173 (proxies API to :8848)
npm run build      # production build (what the Docker image uses)
```

---

## Commit style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(mcp): add relate tool with scope filtering
fix(retrieval): correct MRR calculation for tied ranks
docs(decisions): add ADR-0007 for session-capture design
test(eval): extend golden set with negative cross-project cases
refactor(core): extract dedup logic into standalone function
```

Scope is optional but encouraged for the main packages (`mcp`, `api`, `core`, `ui`, `eval`, `workers`).

---

## Before opening a PR

1. Run the full test suite: `pytest tests/ -v`
2. Run the retrieval eval against the baseline: `python -m scripts.run_eval --baseline synapse/eval/baseline.json`
3. If you changed extraction or retrieval logic, save a new baseline only after confirming the run is better or equal: `python -m scripts.run_eval --save-baseline`
4. Type-check the UI: `cd ui && npm run type-check`

PRs that regress the eval gate (>5% relative drop in hit@k or MRR, or increased violations) will not be merged without an explanation.

---

## What not to contribute

- New managed cloud dependencies — self-hosted only (see R9 in CLAUDE.md).
- Changes to the embedding dimension (1024) — locked at first ingestion; changing it requires re-embedding the entire graph.
- Raw transcript or debug logging as knowledge — the write-pipeline write filters are strict by design.

Questions? Open an issue.
