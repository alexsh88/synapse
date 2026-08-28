"""MCP stdio handshake — verify a wired project by TALKING to its server, not by reading its config.

``ProjectConnector.status()`` answers a different question than the one that matters. It reports
whether ``.mcp.json`` names a "synapse" server and whether the hook command is present. Both stayed
true throughout the period when the server was, in fact, dead in nine of eleven connected projects:
the files were written correctly and the process they described could not start. Claude Code reports
that as ``Failed to reconnect to synapse: -32000`` and nothing else — no exit code, no traceback, no
stderr — so there was never a signal to correct the drift.

This module spawns the exact command the config names, in the project's own working directory, and
speaks the handshake: ``initialize`` -> ``notifications/initialized`` -> ``tools/list``. A server
that cannot start fails here with its real stderr attached, which is the whole point.

The working directory is not incidental. ``python -m`` prepends the CWD to ``sys.path``, so a
project owning a top-level package named like one of ours shadows it and the server dies at import
(acme-sim's own ``synapse/`` client did exactly this). Running the check from anywhere else would
pass while the real thing fails, so ``cwd`` is required rather than defaulted.

Transport framing is newline-delimited JSON, implemented directly rather than through the MCP client
SDK. Two reasons: the check must fail when the SERVER is broken, not when this repo's SDK pin drifts
(the ``mcp.server.fastmcp`` module move already cost three red builds), and a raw reader can report
"the server wrote non-JSON to stdout" — a real failure mode in a codebase where stdout belongs to
the protocol and a stray ``print()`` corrupts it silently.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

# Sent as our preferred version; servers negotiate down and we do not hard-fail on a mismatch.
# A version disagreement is a compatibility question, not a "did it start" question, and this
# module only claims to answer the latter.
PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "synapse-doctor", "version": "1"}

# The server connects Neo4j inside the MCP lifespan, before it answers `initialize`, so this is a
# cold start every time. Measured across the eleven connected projects: 6-8s warm, and 49.5s for
# the largest one with nothing else running. 45s looked generous and was not — it failed a healthy
# project outright. A false alarm is not a harmless error here: the whole reason nine of eleven
# servers stayed dead for weeks is that nobody had a signal worth trusting, and a checker that
# flags working projects gets ignored exactly as fast as one that flags nothing.
DEFAULT_TIMEOUT_S = 120.0

_STDERR_TAIL_LINES = 15
_GRACE_S = 3.0

# Failure kinds. Flat strings rather than an enum: they are printed, logged and compared in tests,
# and every consumer wants the string anyway.
OK = "ok"
SPAWN_FAILED = "spawn-failed"
EXITED = "exited"
TIMEOUT = "timeout"
PROTOCOL_ERROR = "protocol-error"
NO_TOOLS = "no-tools"
NO_CONFIG = "no-config"


@dataclass(frozen=True)
class HandshakeResult:
    """Outcome of one server check. ``ok`` is the only field a caller must look at."""

    ok: bool
    status: str
    detail: str = ""
    tools: tuple[str, ...] = ()
    server_name: str = ""
    server_version: str = ""
    protocol_version: str = ""
    elapsed_s: float = 0.0
    stderr_tail: tuple[str, ...] = ()
    #: Lines the server wrote to stdout that were not JSON-RPC. Non-empty means the transport is
    #: being corrupted even if the handshake happened to survive it.
    stdout_noise: tuple[str, ...] = field(default=())

    def summary(self) -> str:
        """One line, safe to print. Never includes env values — configs carry credentials.

        ASCII only: this is printed to a Windows console whose default code page cannot encode an
        em dash, and the first run rendered every status line with a replacement character.
        """
        if self.ok:
            return f"ok - {len(self.tools)} tools ({self.server_name} {self.server_version})".rstrip()
        return f"{self.status} - {self.detail}" if self.detail else self.status


def _drain(stream: IO[str], sink: deque[str]) -> None:
    """Consume a pipe to EOF. Draining matters even when nobody reads the output: a server that
    fills the stderr pipe buffer blocks on write and then times out here, which would be reported
    as a hang rather than as whatever it was actually trying to say."""
    try:
        for line in stream:
            sink.append(line.rstrip("\n"))
    except (ValueError, OSError):  # pipe closed under us while the process was torn down
        pass


def _pump(stream: IO[str], out: queue.Queue[str | None]) -> None:
    try:
        for line in stream:
            out.put(line)
    except (ValueError, OSError):
        pass
    finally:
        out.put(None)  # EOF sentinel — distinguishes "process gone" from "still thinking"


class _Handshake:
    """One spawned server, driven far enough to prove it works."""

    def __init__(self, proc: subprocess.Popen[str], timeout_s: float) -> None:
        self._proc = proc
        self._deadline = time.monotonic() + timeout_s
        self._stderr: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._noise: list[str] = []
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._eof = False
        assert proc.stdout is not None and proc.stderr is not None
        threading.Thread(target=_pump, args=(proc.stdout, self._lines), daemon=True).start()
        self._err_thread = threading.Thread(target=_drain, args=(proc.stderr, self._stderr), daemon=True)
        self._err_thread.start()

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        # Join first. A server that dies during import writes its traceback and exits, and this
        # property is read immediately afterwards — without the join the reader thread has often
        # not appended the last lines yet, so the one diagnostic worth having comes back empty
        # roughly at random. Bounded, because a still-running server never closes stderr at all.
        self._err_thread.join(timeout=0.5)
        return tuple(self._stderr)

    @property
    def noise(self) -> tuple[str, ...]:
        return tuple(self._noise[:5])

    def _remaining(self) -> float:
        return self._deadline - time.monotonic()

    def send(self, payload: dict[str, Any]) -> str | None:
        """Write one JSON-RPC message. Returns a failure detail, or None on success."""
        stdin = self._proc.stdin
        if stdin is None:
            return "server has no stdin"
        try:
            stdin.write(json.dumps(payload) + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError):
            # The far end is gone. The caller turns this into `exited` with the real stderr, which
            # is more useful than "broken pipe" — the pipe broke because the server died.
            return "server closed stdin"
        return None

    def await_response(self, request_id: int) -> tuple[dict[str, Any] | None, str, str]:
        """Read until the response with ``request_id`` arrives.

        Returns ``(result, status, detail)``. Interleaved notifications and log messages are
        skipped rather than treated as answers — servers legitimately emit them mid-handshake.
        """
        while True:
            remaining = self._remaining()
            if remaining <= 0:
                return None, TIMEOUT, f"no response to request {request_id} within the timeout"
            try:
                line = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if self._proc.poll() is not None and self._eof:
                    break
                continue
            if line is None:
                self._eof = True
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # stdout is the transport; anything non-JSON on it is a bug in the server, usually
                # a print() that should have been a log to stderr. Record and keep reading — the
                # handshake may still complete, and the noise is worth reporting either way.
                self._noise.append(line[:200])
                continue
            if not isinstance(msg, dict):
                self._noise.append(line[:200])
                continue
            if msg.get("id") != request_id:
                continue  # a notification, or an answer to something we did not ask
            if (err := msg.get("error")) is not None:
                code = err.get("code") if isinstance(err, dict) else None
                text = err.get("message") if isinstance(err, dict) else str(err)
                return None, PROTOCOL_ERROR, f"server returned error {code}: {text}"
            result = msg.get("result")
            if not isinstance(result, dict):
                return None, PROTOCOL_ERROR, f"response to request {request_id} carried no result object"
            return result, OK, ""

        code = self._proc.poll()
        return None, EXITED, (f"server exited with code {code} before answering request "
                              f"{request_id}" if code is not None else
                              "server closed stdout before answering")

    def close(self) -> None:
        """Terminate the child. Best-effort and never raises — this runs in a finally block."""
        proc = self._proc
        if proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()          # a well-behaved stdio server exits on EOF
        except OSError:
            pass
        try:
            proc.wait(timeout=_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()


def check_server(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    *,
    cwd: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> HandshakeResult:
    """Spawn an MCP stdio server and complete a handshake against it.

    ``env`` is MERGED over the current environment rather than replacing it, matching what a client
    does: an ``.mcp.json`` names only the few variables the server needs and relies on inheriting
    PATH, SystemRoot and the rest. Replacing the environment would fail servers that work fine.
    """
    started = time.monotonic()
    full_env = {**os.environ, **(env or {})}

    def done(status: str, detail: str = "", **kw: Any) -> HandshakeResult:
        return HandshakeResult(ok=status == OK, status=status, detail=detail,
                               elapsed_s=round(time.monotonic() - started, 2), **kw)

    if not cwd.is_dir():
        return done(SPAWN_FAILED, f"working directory does not exist: {cwd}")

    try:
        proc = subprocess.Popen(
            [command, *(args or [])],
            cwd=str(cwd),
            env=full_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError:
        # By far the most common real cause, and the one `-32000` hides best: the interpreter path
        # baked into .mcp.json does not exist on this machine (moved repo, rebuilt venv, wiring
        # written from inside the container).
        return done(SPAWN_FAILED, f"command not found: {command}")
    except OSError as exc:
        return done(SPAWN_FAILED, f"could not start {command}: {exc}")

    hs = _Handshake(proc, timeout_s)
    try:
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                           "clientInfo": CLIENT_INFO}}
        if (detail := hs.send(init)) is not None:
            return done(EXITED, detail, stderr_tail=hs.stderr_tail)

        result, status, detail = hs.await_response(1)
        if result is None:
            return done(status, detail, stderr_tail=hs.stderr_tail, stdout_noise=hs.noise)

        info = result.get("serverInfo") or {}
        name = str(info.get("name", "")) if isinstance(info, dict) else ""
        version = str(info.get("version", "")) if isinstance(info, dict) else ""
        protocol = str(result.get("protocolVersion", ""))

        if (detail := hs.send({"jsonrpc": "2.0", "method": "notifications/initialized"})) is not None:
            return done(EXITED, detail, stderr_tail=hs.stderr_tail, server_name=name)

        if (detail := hs.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})) is not None:
            return done(EXITED, detail, stderr_tail=hs.stderr_tail, server_name=name)

        listing, status, detail = hs.await_response(2)
        # Annotated: without it mypy infers dict[str, Sequence[str]] from the values and rejects
        # the ** expansion, on the grounds that it could land a tuple in the `detail` string.
        common: dict[str, Any] = {"server_name": name, "server_version": version,
                                  "protocol_version": protocol,
                                  "stderr_tail": hs.stderr_tail, "stdout_noise": hs.noise}
        if listing is None:
            return done(status, detail, **common)

        raw = listing.get("tools")
        tools = tuple(str(t.get("name", "")) for t in raw if isinstance(t, dict)) if isinstance(raw, list) else ()
        if not tools:
            # A server that initializes but exposes nothing is broken in a way the client will not
            # report at all: the connection is "up" and the agent simply has no Synapse tools.
            return done(NO_TOOLS, "handshake succeeded but the server exposes no tools", **common)
        return done(OK, tools=tools, **common)
    finally:
        hs.close()


def _load_servers(folder: Path) -> tuple[dict[str, Any], str]:
    """``(servers, problem)`` — ``problem`` is a human-readable reason the mapping is empty.

    Kept separate from ``read_servers`` so the reason survives. "no synapse server configured" and
    "the file is corrupt JSON" are different problems with different fixes, and collapsing them
    sends you to re-run the wiring when what you needed was to look at the file.
    """
    path = folder / ".mcp.json"
    if not path.is_file():
        return {}, f"no .mcp.json in {folder}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"{path} is not valid JSON: {exc}"
    except OSError as exc:
        return {}, f"{path} could not be read: {exc}"
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return {}, f"{path} has no mcpServers object"
    return servers, ""


def read_servers(folder: Path) -> dict[str, dict[str, Any]]:
    """Return the ``mcpServers`` mapping from ``folder/.mcp.json`` ({} if absent or unreadable)."""
    return _load_servers(folder)[0]


def check_project(
    folder: Path,
    *,
    server_name: str = "synapse",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> HandshakeResult:
    """Check the named server from a project's own ``.mcp.json``, run from that project's folder."""
    if not folder.is_dir():
        return HandshakeResult(ok=False, status=NO_CONFIG, detail=f"project folder not found: {folder}")
    servers, problem = _load_servers(folder)
    entry = servers.get(server_name)
    if not isinstance(entry, dict):
        detail = problem or (f"{folder / '.mcp.json'} does not define an {server_name!r} server")
        return HandshakeResult(ok=False, status=NO_CONFIG, detail=detail)

    command = entry.get("command")
    if not isinstance(command, str) or not command:
        return HandshakeResult(ok=False, status=NO_CONFIG,
                               detail=f"{server_name!r} entry has no command")
    raw_args = entry.get("args") or []
    args = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
    raw_env = entry.get("env") or {}
    env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}
    return check_server(command, args, env, cwd=folder, timeout_s=timeout_s)


@dataclass(frozen=True)
class ProjectCheck:
    """One project's outcome from a sweep."""

    label: str
    folder: Path
    result: HandshakeResult
    #: True if the first attempt timed out under parallel load and this is the serial re-check.
    retried: bool = False


def check_projects(
    targets: Sequence[tuple[str, Path]],
    *,
    server_name: str = "synapse",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    workers: int = 4,
) -> list[ProjectCheck]:
    """Check many projects, then re-check any timeout serially before believing it.

    Parallelism is worth having — eleven cold starts in sequence is minutes — but it manufactures
    the failure it is supposed to detect. Measured: three projects that hand shake in 6-8s alone
    all blew a 45s timeout when four servers started at once, because each one opens its own Neo4j
    session and builds indices. Reported naively that is four dead projects and a wasted morning.

    Only ``timeout`` is retried. ``spawn-failed`` and ``exited`` are deterministic — a missing
    interpreter does not appear under load — so retrying them would just double the runtime of a
    genuinely broken sweep. The retry is surfaced rather than hidden: if a project only passes when
    nothing else is running, that is worth seeing.

    ``workers=1`` means "serial, and mean it": no retry, because the caller asked for the slow
    honest sweep and a timeout under no contention is an answer rather than a fluke. The decision
    keys off the REQUESTED worker count, not the clamped one — a single target checked in parallel
    mode still deserves the retry, since the load that matters may come from outside this process.
    """
    if not targets:
        return []
    retry_timeouts = workers > 1

    def run(item: tuple[str, Path]) -> ProjectCheck:
        label, folder = item
        return ProjectCheck(label, folder,
                            check_project(folder, server_name=server_name, timeout_s=timeout_s))

    n = max(1, min(workers, len(targets)))
    if n == 1:
        checks = [run(item) for item in targets]
    else:
        with ThreadPoolExecutor(max_workers=n) as pool:
            checks = list(pool.map(run, targets))

    if retry_timeouts:
        for i, check in enumerate(checks):
            if check.result.status == TIMEOUT:
                again = check_project(check.folder, server_name=server_name, timeout_s=timeout_s)
                checks[i] = ProjectCheck(check.label, check.folder, again, retried=True)
    return checks
