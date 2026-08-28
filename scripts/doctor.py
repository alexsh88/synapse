"""synapse doctor — prove each connected project's MCP server actually starts.

    python -m scripts.doctor                     # every registered project
    python -m scripts.doctor acme-jobs acme-flow   # by registry id
    python -m scripts.doctor C:/dev/some-project # or by path, no registry needed
    python -m scripts.doctor . --json            # machine-readable, for CI

Existing tooling checks that the wiring FILES are right. `wire_project --list` reads each
project's `.mcp.json` for a "synapse" key; the Projects page shows the same thing. That check was
green in every project during the stretch when the server was dead in nine of eleven of them,
because writing the config and being able to run it are different facts. Claude Code renders the
gap as `Failed to reconnect to synapse: -32000` — no exit code, no stderr, nothing to act on.

This spawns the exact command each config names, from that project's own folder, and completes the
MCP handshake. Failures come back with the real reason attached. Exit status is 1 if any project
fails, so it can gate a rollout instead of being read by eye.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from synapse.core import mcp_health as mh

REPO = Path(__file__).resolve().parents[1]


def _registry() -> dict[str, dict]:
    """Project registry, or {} if it cannot be loaded.

    Deliberately forgiving: importing the registry pulls in ``settings``, which can fail on a
    machine whose configuration is exactly what you are trying to diagnose. A doctor that refuses
    to run when things are broken is no doctor at all — fall back to path arguments.
    """
    try:
        from synapse.core import registry
        return registry.all_projects()
    except Exception as exc:  # noqa: BLE001 — any config failure degrades to path-only mode
        print(f"[warn] project registry unavailable ({type(exc).__name__}: {exc}); "
              f"pass project paths explicitly", file=sys.stderr)
        return {}


def _targets(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Resolve arguments to ``(label, folder)`` pairs. An argument is a registry id or a path."""
    projects = _registry()
    if not args.target:
        if projects:
            return [(pid, Path(p["path"])) for pid, p in projects.items()]
        print("[warn] no registry and no targets given — checking the current directory",
              file=sys.stderr)
        return [(Path.cwd().name, Path.cwd())]

    out: list[tuple[str, Path]] = []
    for raw in args.target:
        if raw in projects:
            out.append((raw, Path(projects[raw]["path"])))
        else:
            folder = Path(raw).resolve()
            # An unknown bare word is far more likely a typo'd id than a relative folder that
            # happens not to exist, and saying so beats "project folder not found".
            if not folder.is_dir() and "/" not in raw and "\\" not in raw and projects:
                print(f"[error] unknown project {raw!r}. Known: {', '.join(sorted(projects))}",
                      file=sys.stderr)
                raise SystemExit(2)
            out.append((folder.name or raw, folder))
    return out


def _render(check: mh.ProjectCheck, *, width: int) -> None:
    res = check.result
    mark = "PASS" if res.ok else "FAIL"
    tools = str(len(res.tools)) if res.ok else "-"
    note = "  (retried serially)" if check.retried else ""
    print(f"  {mark}  {check.label:<{width}}  {tools:>3}  {res.elapsed_s:>5.1f}s  {res.summary()}{note}")
    if not res.ok and res.stderr_tail:
        print("        server stderr (last lines):")
        for line in res.stderr_tail:
            print(f"        | {line}")
    if res.stdout_noise:
        # Worth showing even on a pass: stdout belongs to the protocol, so anything else on it is
        # a print() that will eventually land mid-message and break a live session.
        print("        stray stdout (should be logged to stderr):")
        for line in res.stdout_noise:
            print(f"        | {line}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify each project's Synapse MCP server by handshake.")
    ap.add_argument("target", nargs="*", help="registry ids or project paths (default: all registered)")
    ap.add_argument("--server", default="synapse", help="server name inside .mcp.json (default: synapse)")
    ap.add_argument("--timeout", type=float, default=mh.DEFAULT_TIMEOUT_S,
                    help=f"seconds to wait per project (default: {mh.DEFAULT_TIMEOUT_S:g})")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel checks (default: 4; 1 = serial). Timeouts are always "
                         "re-checked serially before being reported.")
    ap.add_argument("--json", action="store_true", dest="as_json", help="emit JSON instead of a table")
    args = ap.parse_args()

    targets = _targets(args)
    if not targets:
        print("nothing to check.")
        return 0

    checks = mh.check_projects(targets, server_name=args.server,
                               timeout_s=args.timeout, workers=args.workers)
    failed = [c for c in checks if not c.result.ok]

    if args.as_json:
        print(json.dumps([{
            "project": c.label, "path": str(c.folder), "ok": c.result.ok,
            "status": c.result.status, "detail": c.result.detail, "tools": list(c.result.tools),
            "elapsed_s": c.result.elapsed_s, "retried": c.retried,
            "stderr_tail": list(c.result.stderr_tail),
            "stdout_noise": list(c.result.stdout_noise),
        } for c in checks], indent=2))
        return 1 if failed else 0

    width = max(len(c.label) for c in checks)
    print(f"== MCP handshake: {len(checks)} project(s), server {args.server!r} ==")
    for check in checks:
        _render(check, width=width)

    print(f"\n{len(checks) - len(failed)}/{len(checks)} healthy.")
    if slow := [c.label for c in checks if c.retried and c.result.ok]:
        print(f"[note] only passed once nothing else was starting: {', '.join(slow)} "
              f"(healthy, but slow to come up)")
    if failed:
        print("\nFailing: " + ", ".join(c.label for c in failed))
        statuses = {c.result.status for c in failed}
        if statuses & {mh.SPAWN_FAILED, mh.NO_CONFIG}:
            print("  Wiring looks wrong - re-run:  python -m scripts.wire_project <id>")
        if statuses & {mh.EXITED, mh.TIMEOUT}:
            print("  The server starts and dies - check the stderr above, then that Neo4j is up "
                  "(docker compose ps) and that this repo's .venv still exists.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
