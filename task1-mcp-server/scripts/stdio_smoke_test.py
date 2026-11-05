#!/usr/bin/env python3
"""Drive the customer MCP server over stdio with raw JSON-RPC frames.

Starts the server as a subprocess, plays a scripted conversation that covers
the success path, every validation failure class, business errors and
malformed input, prints each exchange, and finally proves that stdout carried
nothing except JSON-RPC frames. Only the standard library is needed.

    python scripts/stdio_smoke_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from customer_mcp_server.testing import StdioServerHarness


def describe(response: dict[str, Any]) -> str:
    if "error" in response:
        err = response["error"]
        data = err.get("data")
        extra = ""
        if isinstance(data, dict) and "issues" in data:
            extra = " issues=" + ", ".join(f"{i['path']}:{i['type']}" for i in data["issues"])
        return f"ERROR {err['code']} {err['message']!r}{extra}"
    result = response["result"]
    if result.get("isError"):
        return f"TOOL ERROR {result['structuredContent']['error']}"
    if "tools" in result:
        return "TOOLS " + ", ".join(t["name"] for t in result["tools"])
    if "serverInfo" in result:
        info = result["serverInfo"]
        return f"INITIALIZED {info['name']} {info['version']} protocol={result['protocolVersion']}"
    if "structuredContent" in result:
        return f"RESULT {json.dumps(result['structuredContent'])}"
    return f"RESULT {json.dumps(result)}"


def step(title: str, response: dict[str, Any] | None) -> None:
    print(f"\n# {title}")
    if response is not None:
        print("  ->", describe(response))


def refund(server: StdioServerHarness, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "customer_id": "CUST-00001",
        "amount": 25.5,
        "reason": "Damaged item returned by customer",
    }
    arguments.update(overrides)
    return server.call_tool("trigger_refund", arguments)


def main() -> int:
    server = StdioServerHarness(log_level="DEBUG")
    try:
        step("initialize", server.initialize())
        step("notifications/initialized (no response expected)", None)
        step("tools/list", server.request("tools/list"))

        step(
            "get_customer_record: valid id",
            server.call_tool("get_customer_record", {"customer_id": "CUST-00001"}),
        )
        step(
            "get_customer_record: malformed id",
            server.call_tool("get_customer_record", {"customer_id": "cust-1"}),
        )
        step("get_customer_record: missing arguments", server.call_tool("get_customer_record"))

        step(
            "trigger_refund: wrong type and short reason",
            refund(server, amount="25", reason="short"),
        )
        step("trigger_refund: negative amount", refund(server, amount=-5))
        step("trigger_refund: more than two decimal places", refund(server, amount=10.005))
        step("trigger_refund: whitespace padded reason", refund(server, reason="      ok      "))
        step("trigger_refund: unknown extra field", refund(server, approved_by="me"))
        step("trigger_refund: valid", refund(server))
        step(
            "trigger_refund: unknown customer (tool error, not protocol error)",
            refund(server, customer_id="CUST-99999"),
        )
        step(
            "trigger_refund: exceeds balance (tool error)",
            refund(server, customer_id="CUST-00002", amount=500),
        )
        step(
            "trigger_refund: suspended customer (tool error)",
            refund(server, customer_id="CUST-00003", amount=5),
        )

        step("unknown tool", server.call_tool("delete_everything", {}))
        step(
            "arguments that are not an object",
            server.call_tool("get_customer_record", ["CUST-00001"]),
        )

        server.send_raw("this is not json")
        step("malformed frame (invalid JSON)", server.read_frame())
        server.send_raw(json.dumps({"hello": "world"}))
        step("malformed frame (JSON but not JSON-RPC)", server.read_frame())

        step("ping (server still healthy)", server.request("ping"))
    finally:
        exit_code = server.close()

    stray = server.stray_stdout_lines()
    print("\n# Verification")
    print(f"  stdout frames : {len(server.stdout_lines)}")
    print(f"  stray lines   : {len(stray)}")
    print(f"  stderr lines  : {len(server.stderr_lines)} (JSON logs)")
    print(f"  exit code     : {exit_code}")
    if server.stderr_lines:
        print("  first log     :", server.stderr_lines[0][:160])
    ok = not stray and exit_code == 0
    print("\nSTDOUT PURITY:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
