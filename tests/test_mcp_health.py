"""Handshake checker tests — driven against a fake stdio server, not the real one.

Everything here spawns ``sys.executable`` running a script written into ``tmp_path``, so the suite
needs no Neo4j, no Ollama and no venv layout: it runs identically on a developer's Windows box and
on the Linux CI runner. That is the point — a health check that only works on the machine that is
already healthy is the failure mode this module was written to remove.

The fake server's behaviour is selected by ``FAKE_MODE`` in its environment, which doubles as the
test that ``check_server`` merges env correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from synapse.core import mcp_health as mh

# Raw string on purpose: the template is Python source, and its own escapes (\n) must survive
# being a literal in THIS file rather than being interpreted here.
_FAKE_SERVER = r'''
import json, os, sys, time

mode = os.environ.get("FAKE_MODE", "ok")

if mode == "crash":
    sys.stderr.write("Traceback (most recent call last):\n")
    sys.stderr.write("ModuleNotFoundError: No module named 'synapse.mcp.server'\n")
    sys.stderr.flush()
    raise SystemExit(1)

if mode == "hang":
    time.sleep(60)
    raise SystemExit(0)

# Hangs on its first invocation and serves normally afterwards — stands in for a healthy server
# that misses the deadline only because three others are starting at the same time.
if mode == "flaky":
    marker = os.environ["FAKE_MARKER"]
    if not os.path.exists(marker):
        open(marker, "w").close()
        time.sleep(60)
        raise SystemExit(0)


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


if mode == "noise":
    sys.stdout.write("loading embedder weights...\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        if mode == "error":
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": "lifespan failed"}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            # The cwd rides out in `version` so a test can prove the server really was launched
            # from the project folder — the shadowed-package failure only reproduces there.
            "serverInfo": {"name": "fake-synapse", "version": os.getcwd()},
        }})
    elif method == "tools/list":
        names = [] if mode == "notools" else ["remember", "recall", "brief"]
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{"name": n} for n in names]}})
'''


@pytest.fixture
def fake_server(tmp_path: Path) -> Path:
    path = tmp_path / "fake_server.py"
    path.write_text(_FAKE_SERVER, encoding="utf-8")
    return path


def _check(fake: Path, mode: str = "ok", *, cwd: Path | None = None, timeout: float = 20.0):
    return mh.check_server(sys.executable, [str(fake)], {"FAKE_MODE": mode},
                           cwd=cwd or fake.parent, timeout_s=timeout)


# --- the happy path ---------------------------------------------------------------------------

def test_healthy_server_reports_its_tools(fake_server: Path):
    res = _check(fake_server)
    assert res.ok and res.status == mh.OK
    assert res.tools == ("remember", "recall", "brief")
    assert res.server_name == "fake-synapse"
    assert res.protocol_version == "2025-06-18"
    assert res.stdout_noise == ()


def test_server_runs_in_the_project_folder(fake_server: Path, tmp_path: Path):
    """The check must reproduce the import environment, not merely a working one.

    `python -m` puts the CWD on sys.path, so a project that owns a top-level package named like
    one of ours shadows it and the server dies at import. Checking from anywhere else would pass
    while the real session fails.
    """
    project = tmp_path / "acme-sim"
    project.mkdir()
    res = _check(fake_server, cwd=project)
    assert res.ok
    assert Path(res.server_version).resolve() == project.resolve()


def test_env_is_merged_not_replaced(fake_server: Path):
    """A .mcp.json names two or three variables and inherits the rest; replacing the environment
    would break servers that work — PATH and SystemRoot are not optional."""
    res = _check(fake_server, "ok")
    assert res.ok, res.summary()   # the interpreter itself needs the inherited environment


# --- the failures worth telling apart ---------------------------------------------------------

def test_missing_interpreter_is_named(tmp_path: Path):
    """The most common real cause of -32000: the venv path in .mcp.json is not on this machine."""
    res = mh.check_server("definitely-not-a-real-binary-9f3c", [], {}, cwd=tmp_path, timeout_s=5)
    assert not res.ok and res.status == mh.SPAWN_FAILED
    assert "definitely-not-a-real-binary-9f3c" in res.detail


def test_crash_at_import_surfaces_the_traceback(fake_server: Path):
    res = _check(fake_server, "crash", timeout=10)
    assert not res.ok and res.status == mh.EXITED
    assert any("ModuleNotFoundError" in line for line in res.stderr_tail), res.stderr_tail


def test_hanging_server_times_out(fake_server: Path):
    res = _check(fake_server, "hang", timeout=1.5)
    assert not res.ok and res.status == mh.TIMEOUT
    assert res.elapsed_s >= 1.0


def test_error_response_is_protocol_error(fake_server: Path):
    res = _check(fake_server, "error", timeout=10)
    assert not res.ok and res.status == mh.PROTOCOL_ERROR
    assert "-32603" in res.detail and "lifespan failed" in res.detail


def test_server_with_no_tools_is_not_healthy(fake_server: Path):
    """Initialising is not the same as working. A client shows this connection as up while the
    agent silently has no Synapse tools at all — the quietest failure of the set."""
    res = _check(fake_server, "notools", timeout=10)
    assert not res.ok and res.status == mh.NO_TOOLS


def test_stray_stdout_is_reported_but_not_fatal(fake_server: Path):
    """stdout is the transport. A print() that should have been a log corrupts it, so it gets
    reported even when the handshake survives."""
    res = _check(fake_server, "noise", timeout=10)
    assert res.ok
    assert res.stdout_noise and "loading embedder weights" in res.stdout_noise[0]


# --- reading the project's own config ---------------------------------------------------------

def _write_mcp(folder: Path, entry: dict | None, *, raw: str | None = None) -> None:
    body = raw if raw is not None else json.dumps(
        {"mcpServers": {"synapse": entry}} if entry else {"mcpServers": {}}, indent=2)
    (folder / ".mcp.json").write_text(body, encoding="utf-8")


def test_check_project_end_to_end(fake_server: Path, tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    _write_mcp(project, {"command": sys.executable, "args": [str(fake_server)],
                         "env": {"FAKE_MODE": "ok"}})
    res = mh.check_project(project, timeout_s=20)
    assert res.ok and res.tools == ("remember", "recall", "brief")


def test_check_project_missing_config(tmp_path: Path):
    res = mh.check_project(tmp_path)
    assert not res.ok and res.status == mh.NO_CONFIG and "no .mcp.json" in res.detail


def test_check_project_config_without_synapse(tmp_path: Path):
    _write_mcp(tmp_path, None)
    res = mh.check_project(tmp_path)
    assert not res.ok and res.status == mh.NO_CONFIG and "synapse" in res.detail


def test_check_project_corrupt_json_says_so(tmp_path: Path):
    """Distinct from 'not configured' — the fix is to look at the file, not to re-run wiring."""
    _write_mcp(tmp_path, None, raw="{ this is not json")
    res = mh.check_project(tmp_path)
    assert not res.ok and res.status == mh.NO_CONFIG and "not valid JSON" in res.detail


def test_check_project_entry_without_command(tmp_path: Path):
    _write_mcp(tmp_path, {"args": ["-m", "synapse.mcp.server"]})
    res = mh.check_project(tmp_path)
    assert not res.ok and res.status == mh.NO_CONFIG and "no command" in res.detail


def test_check_project_missing_folder(tmp_path: Path):
    res = mh.check_project(tmp_path / "nope")
    assert not res.ok and res.status == mh.NO_CONFIG


def test_read_servers_is_empty_on_corrupt_config(tmp_path: Path):
    _write_mcp(tmp_path, None, raw="{ nope")
    assert mh.read_servers(tmp_path) == {}


# --- sweeping several projects ----------------------------------------------------------------

def _project(root: Path, name: str, fake: Path, mode: str, **env: str) -> tuple[str, Path]:
    folder = root / name
    folder.mkdir()
    _write_mcp(folder, {"command": sys.executable, "args": [str(fake)],
                        "env": {"FAKE_MODE": mode, **env}})
    return name, folder


def test_sweep_checks_every_project(fake_server: Path, tmp_path: Path):
    targets = [_project(tmp_path, "good", fake_server, "ok"),
               _project(tmp_path, "bad", fake_server, "crash")]
    checks = mh.check_projects(targets, timeout_s=20, workers=2)
    by_label = {c.label: c for c in checks}
    assert by_label["good"].result.ok
    assert not by_label["bad"].result.ok
    assert by_label["bad"].result.status == mh.EXITED


def test_timeout_under_load_is_rechecked_before_being_believed(fake_server: Path, tmp_path: Path):
    """The false alarm this exists to prevent.

    Three healthy projects that handshake in under 8s alone all blew a 45s timeout when four
    servers started at once — reported naively, that was four dead projects that were all fine.
    A server that misses the deadline once and answers on a quiet retry must come back healthy.
    """
    target = _project(tmp_path, "slow", fake_server, "flaky",
                      FAKE_MARKER=str(tmp_path / "seen.marker"))
    checks = mh.check_projects([target], timeout_s=2.0, workers=2)
    assert len(checks) == 1
    assert checks[0].result.ok, checks[0].result.summary()
    assert checks[0].retried, "a passing re-check must be visible, not silently swallowed"


def test_serial_sweep_does_not_retry(fake_server: Path, tmp_path: Path):
    """With one worker there is no contention to blame, so a timeout is the answer, not a fluke.
    Retrying anyway would double the runtime of a genuinely broken sweep."""
    target = _project(tmp_path, "stuck", fake_server, "hang")
    checks = mh.check_projects([target], timeout_s=1.5, workers=1)
    assert checks[0].result.status == mh.TIMEOUT
    assert not checks[0].retried


def test_deterministic_failures_are_not_retried(fake_server: Path, tmp_path: Path):
    """A missing interpreter does not become present under lighter load."""
    folder = tmp_path / "gone"
    folder.mkdir()
    _write_mcp(folder, {"command": "definitely-not-a-real-binary-9f3c", "args": []})
    checks = mh.check_projects([("gone", folder)], timeout_s=5, workers=2)
    assert checks[0].result.status == mh.SPAWN_FAILED
    assert not checks[0].retried


def test_sweep_of_nothing(tmp_path: Path):
    assert mh.check_projects([]) == []
