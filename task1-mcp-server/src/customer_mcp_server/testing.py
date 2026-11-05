"""Test support: drive the server as a real subprocess over stdio.

Shared by the automated tests and ``scripts/stdio_smoke_test.py`` so both
exercise the exact pipes an agent host would use. Only the standard library
is required.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
READ_TIMEOUT_SECONDS = 15


def subprocess_environment() -> dict[str, str]:
    """Environment for a child interpreter that can import this package from source."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC_DIR), env.get("PYTHONPATH", "")) if p)
    return env


def is_jsonrpc_frame(line: str) -> bool:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and obj.get("jsonrpc") == "2.0"


class StdioServerHarness:
    """Talk raw JSON-RPC to ``python -m customer_mcp_server`` through its pipes."""

    def __init__(self, log_level: str = "DEBUG", *, command: list[str] | None = None) -> None:
        """Spawn the server; ``command`` replaces the default module invocation (tests only)."""
        self.proc = subprocess.Popen(
            command or [sys.executable, "-m", "customer_mcp_server", "--log-level", log_level],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=subprocess_environment(),
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._threads = [
            threading.Thread(target=self._pump_stdout, daemon=True),
            threading.Thread(target=self._pump_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._next_id = 1

    # -- plumbing ------------------------------------------------------------

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.stdout_lines.append(line.rstrip("\n"))
            self._stdout_queue.put(line)
        self._stdout_queue.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip("\n"))

    def send_raw(self, line: str) -> None:
        """Write one raw line to the server's stdin (no validation on purpose)."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_frame(self) -> dict[str, Any]:
        line = self._stdout_queue.get(timeout=READ_TIMEOUT_SECONDS)
        if line is None:
            raise RuntimeError("server closed stdout before answering")
        frame: dict[str, Any] = json.loads(line)
        return frame

    # -- JSON-RPC helpers ----------------------------------------------------

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: int | str | None = None,
    ) -> dict[str, Any]:
        if request_id is None:
            request_id = self._next_id
            self._next_id += 1
        frame: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            frame["params"] = params
        self.send_raw(json.dumps(frame))
        response = self.read_frame()
        if response.get("id") != request_id:
            raise RuntimeError(f"response id mismatch: {response}")
        return response

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        frame: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        self.send_raw(json.dumps(frame))

    def initialize(self, protocol_version: str = "2025-06-18") -> dict[str, Any]:
        response = self.request(
            "initialize",
            {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "stdio-harness", "version": "1.0.0"},
            },
        )
        self.notify("notifications/initialized")
        return response

    def call_tool(self, name: str, arguments: Any | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return self.request("tools/call", params)

    # -- lifecycle and auditing ---------------------------------------------

    def stray_stdout_lines(self) -> list[str]:
        """Lines seen on stdout that are not JSON-RPC 2.0 frames (expected: none)."""
        return [line for line in self.stdout_lines if not is_jsonrpc_frame(line)]

    def close(self) -> int:
        """Close stdin, wait for a clean exit and return the exit code."""
        assert self.proc.stdin is not None
        if self.proc.poll() is None:
            self.proc.stdin.close()
        try:
            code = self.proc.wait(timeout=READ_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait(timeout=5)
        for thread in self._threads:
            thread.join(timeout=5)
        return code
