"""End-to-end tests over the real stdio transport using raw JSON-RPC frames.

These tests spawn ``python -m customer_mcp_server`` and speak to it through
its actual pipes, so they verify the three evaluation criteria directly:
stdout isolation, protocol compliance and validation behaviour.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import textwrap
import time
from typing import Any

import pytest

from customer_mcp_server.testing import StdioServerHarness

pytestmark = pytest.mark.integration

INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
PARSE_ERROR = -32700
INVALID_REQUEST = -32600

VALID_REFUND = {
    "customer_id": "CUST-00001",
    "amount": 25.5,
    "reason": "Damaged item returned by customer",
}


def test_handshake_and_tool_listing(harness: StdioServerHarness) -> None:
    init = harness.initialize()
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "customer-mcp-server"
    assert "tools" in init["result"]["capabilities"]

    listing = harness.request("tools/list")
    tools = {tool["name"]: tool for tool in listing["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}

    record_schema = tools["get_customer_record"]["inputSchema"]
    assert record_schema["type"] == "object"
    assert record_schema["additionalProperties"] is False
    assert record_schema["required"] == ["customer_id"]
    assert record_schema["properties"]["customer_id"]["pattern"] == r"^CUST-[0-9]{5}$"

    refund_schema = tools["trigger_refund"]["inputSchema"]
    assert set(refund_schema["required"]) == {"customer_id", "amount", "reason"}
    assert refund_schema["properties"]["amount"]["exclusiveMinimum"] == 0
    assert refund_schema["properties"]["reason"]["minLength"] == 10
    assert tools["trigger_refund"]["annotations"]["destructiveHint"] is True
    assert tools["get_customer_record"]["annotations"]["readOnlyHint"] is True


def test_successful_tool_calls_return_structured_content(harness: StdioServerHarness) -> None:
    harness.initialize()
    record = harness.call_tool("get_customer_record", {"customer_id": "CUST-00001"})
    result = record["result"]
    assert result.get("isError", False) is False
    assert result["structuredContent"]["customer_id"] == "CUST-00001"
    assert result["content"][0]["type"] == "text"

    refund = harness.call_tool("trigger_refund", VALID_REFUND)
    receipt = refund["result"]["structuredContent"]
    assert receipt["status"] == "accepted"
    assert receipt["amount"] == 25.5
    assert receipt["remaining_balance"] == 224.5


@pytest.mark.parametrize(
    ("tool", "arguments", "expected_paths"),
    [
        ("get_customer_record", {"customer_id": "cust-1"}, {"customer_id"}),
        ("get_customer_record", {}, {"customer_id"}),
        ("get_customer_record", None, {"customer_id"}),
        ("get_customer_record", {"customer_id": 1}, {"customer_id"}),
        ("get_customer_record", {"customer_id": "CUST-00001", "x": 1}, {"x"}),
        ("trigger_refund", {**VALID_REFUND, "amount": "25"}, {"amount"}),
        ("trigger_refund", {**VALID_REFUND, "amount": 0}, {"amount"}),
        ("trigger_refund", {**VALID_REFUND, "amount": -1}, {"amount"}),
        ("trigger_refund", {**VALID_REFUND, "amount": 10.005}, {"amount"}),
        ("trigger_refund", {**VALID_REFUND, "reason": "short"}, {"reason"}),
        ("trigger_refund", {**VALID_REFUND, "reason": "          "}, {"reason"}),
        ("trigger_refund", {"customer_id": "CUST-00001"}, {"amount", "reason"}),
        (
            "trigger_refund",
            {"customer_id": "bad", "amount": "x", "reason": "no"},
            {"customer_id", "amount", "reason"},
        ),
    ],
)
def test_invalid_arguments_map_to_invalid_params(
    harness: StdioServerHarness,
    tool: str,
    arguments: dict[str, Any] | None,
    expected_paths: set[str],
) -> None:
    harness.initialize()
    response = harness.call_tool(tool, arguments)
    assert "result" not in response
    error = response["error"]
    assert error["code"] == INVALID_PARAMS
    assert error["data"]["tool"] == tool
    reported_paths = {issue["path"] for issue in error["data"]["issues"]}
    assert reported_paths == expected_paths
    # Submitted values are never echoed back: issues carry only path, type and message.
    assert all(set(issue) == {"path", "type", "message"} for issue in error["data"]["issues"])


def test_unknown_tool_is_invalid_params(harness: StdioServerHarness) -> None:
    harness.initialize()
    response = harness.call_tool("delete_everything", {})
    assert response["error"]["code"] == INVALID_PARAMS
    assert response["error"]["message"] == "Unknown tool: delete_everything"
    assert set(response["error"]["data"]["available_tools"]) == {
        "get_customer_record",
        "trigger_refund",
    }


def test_non_object_arguments_are_rejected_at_the_protocol_layer(
    harness: StdioServerHarness,
) -> None:
    harness.initialize()
    response = harness.call_tool("get_customer_record", ["CUST-00001"])
    assert response["error"]["code"] == INVALID_PARAMS


def test_business_errors_are_tool_results_not_protocol_errors(
    harness: StdioServerHarness,
) -> None:
    harness.initialize()
    cases = {
        "customer_not_found": {**VALID_REFUND, "customer_id": "CUST-99999"},
        "refund_exceeds_balance": {**VALID_REFUND, "customer_id": "CUST-00002", "amount": 500},
        "customer_not_eligible": {**VALID_REFUND, "customer_id": "CUST-00003"},
    }
    for expected_code, arguments in cases.items():
        response = harness.call_tool("trigger_refund", arguments)
        assert "error" not in response, response
        result = response["result"]
        assert result["isError"] is True
        assert result["structuredContent"]["error"]["code"] == expected_code
        assert result["content"][0]["text"].startswith(expected_code)


def test_malformed_frames_are_answered_and_server_survives(harness: StdioServerHarness) -> None:
    harness.initialize()

    harness.send_raw("this is not json")
    parse_error = harness.read_frame()
    assert parse_error["id"] is None
    assert parse_error["error"]["code"] == PARSE_ERROR

    harness.send_raw('{"hello": "world"}')
    invalid_request = harness.read_frame()
    assert invalid_request["id"] is None
    assert invalid_request["error"]["code"] == INVALID_REQUEST

    assert harness.request("ping")["result"] == {}


def test_request_ids_are_echoed_for_strings_and_zero(harness: StdioServerHarness) -> None:
    harness.initialize()
    assert harness.request("ping", request_id="req-abc")["id"] == "req-abc"
    assert harness.request("ping", request_id=0)["id"] == 0


def test_stdout_carries_only_jsonrpc_frames(harness: StdioServerHarness) -> None:
    """Exercise every code path with DEBUG logging on, then audit the streams."""
    harness.initialize()
    harness.request("tools/list")
    harness.call_tool("get_customer_record", {"customer_id": "CUST-00001"})
    harness.call_tool("get_customer_record", {"customer_id": "nope"})
    harness.call_tool("trigger_refund", VALID_REFUND)
    harness.call_tool("trigger_refund", {**VALID_REFUND, "customer_id": "CUST-99999"})
    harness.call_tool("missing_tool")
    harness.send_raw("garbage")
    harness.read_frame()
    harness.request("ping")

    exit_code = harness.close()

    assert exit_code == 0
    assert harness.stdout_lines, "expected JSON-RPC responses on stdout"
    assert harness.stray_stdout_lines() == []
    assert harness.stderr_lines, "diagnostics must be written to stderr"
    assert any('"event": "server.start"' in line for line in harness.stderr_lines)
    assert any('"event": "tools.call.invalid_params"' in line for line in harness.stderr_lines)


@pytest.mark.parametrize("bad_id", ["1.5", "true", "null", "[1]", '{"a": 1}'])
def test_requests_with_malformed_ids_are_answered_not_dropped(
    harness: StdioServerHarness, bad_id: str
) -> None:
    harness.initialize()
    harness.send_raw('{"jsonrpc": "2.0", "id": ' + bad_id + ', "method": "ping"}')
    response = harness.read_frame()
    assert response["id"] is None
    assert response["error"]["code"] == INVALID_REQUEST
    assert harness.request("ping")["result"] == {}  # the session is unaffected


def test_invalid_requests_echo_a_recoverable_id(harness: StdioServerHarness) -> None:
    harness.initialize()
    harness.send_raw('{"jsonrpc": "2.0", "id": 77, "method": "ping", "params": []}')
    response = harness.read_frame()
    assert response["id"] == 77
    assert response["error"]["code"] == INVALID_REQUEST
    harness.send_raw('{"jsonrpc": "2.0", "id": "abc", "method": 5}')
    assert harness.read_frame()["id"] == "abc"


def test_error_responses_are_bounded_for_oversized_input(harness: StdioServerHarness) -> None:
    harness.initialize()
    huge_name = "x" * 100_000
    unknown = harness.call_tool(huge_name)
    assert unknown["error"]["code"] == INVALID_PARAMS
    assert len(json.dumps(unknown)) < 2_000
    assert unknown["error"]["data"]["tool"].endswith("...")

    arguments = {f"field{i}": i for i in range(25)}  # within the member cap
    arguments.update(VALID_REFUND)
    invalid = harness.call_tool("trigger_refund", arguments)
    data = invalid["error"]["data"]
    assert len(data["issues"]) == 20
    assert data["omitted_issues"] == 5
    assert len(json.dumps(invalid)) < 10_000


def test_stray_prints_during_a_session_never_reach_stdout() -> None:
    """An unflushed print is block-buffered and would otherwise reach the wire at exit."""
    program = textwrap.dedent(
        """
        from customer_mcp_server import __main__ as entry
        from customer_mcp_server import service

        original = service.CustomerService.get_customer_record

        def noisy(self, customer_id):
            print("STRAY PRINT FROM A LIBRARY", end="")
            print("SECOND STRAY LINE")
            return original(self, customer_id)

        service.CustomerService.get_customer_record = noisy
        entry.run()
        """
    )
    server = StdioServerHarness(command=[sys.executable, "-c", program, "--log-level", "DEBUG"])
    try:
        server.initialize()
        record = server.call_tool("get_customer_record", {"customer_id": "CUST-00001"})
        assert record["result"]["structuredContent"]["customer_id"] == "CUST-00001"
    finally:
        exit_code = server.close()
    assert exit_code == 0
    assert server.stray_stdout_lines() == []
    assert any("STRAY PRINT FROM A LIBRARY" in line for line in server.stderr_lines)


@pytest.mark.skipif(os.name != "posix", reason="128 + signal exit codes are a POSIX convention")
@pytest.mark.parametrize(("signum", "expected_code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_termination_signals_exit_promptly_with_pure_stdout(
    signum: signal.Signals, expected_code: int
) -> None:
    server = StdioServerHarness()
    try:
        server.initialize()
        started = time.perf_counter()
        server.proc.send_signal(signum)
        code = server.proc.wait(timeout=5)
        elapsed = time.perf_counter() - started
    finally:
        server.close()
    assert code == expected_code
    assert elapsed < 5
    assert server.stray_stdout_lines() == []
    assert any('"event": "server.signal"' in line for line in server.stderr_lines)


def test_a_refund_answered_while_the_client_closes_stdin_is_never_lost() -> None:
    """Pipe style hosts send the last frame and close stdin at once; the receipt must arrive."""
    frames = [
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pipe", "version": "1"},
                },
            }
        ),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "trigger_refund", "arguments": VALID_REFUND},
            }
        ),
    ]
    import subprocess

    from customer_mcp_server.testing import subprocess_environment

    for _ in range(12):
        completed = subprocess.run(
            [sys.executable, "-m", "customer_mcp_server", "--log-level", "INFO"],
            input=("\n".join(frames) + "\n").encode(),
            capture_output=True,
            env=subprocess_environment(),
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0
        answers = [json.loads(line) for line in completed.stdout.decode().splitlines()]
        ids = [a.get("id") for a in answers]
        executed = b'"event": "refund.accepted"' in completed.stderr
        assert (2 in ids) == executed, (ids, completed.stderr[-500:])
        assert 2 in ids  # the refund both executed and was answered


def test_oversized_frames_are_discarded_without_parsing(harness: StdioServerHarness) -> None:
    harness.initialize()
    huge = '{"jsonrpc": "2.0", "id": 9, "method": "ping", "params": {"pad": "'
    huge += "x" * (5 * 1024 * 1024) + '"}}'
    started = time.perf_counter()
    harness.send_raw(huge)
    response = harness.read_frame()
    assert time.perf_counter() - started < 5
    assert response["id"] is None
    assert response["error"]["code"] == INVALID_REQUEST
    assert "exceeds" in response["error"]["message"]
    assert harness.request("ping")["result"] == {}


def test_hostile_argument_shapes_are_rejected_cheaply(harness: StdioServerHarness) -> None:
    harness.initialize()
    started = time.perf_counter()
    many_keys = {f"k{i}": i for i in range(50_000)}
    many_keys.update(VALID_REFUND)
    rejected = harness.call_tool("trigger_refund", many_keys)
    assert rejected["error"]["code"] == INVALID_PARAMS
    assert rejected["error"]["data"]["issues"][0]["type"] == "too_many_fields"
    long_reason = harness.call_tool("trigger_refund", {**VALID_REFUND, "reason": "a" * 2_000_000})
    assert long_reason["error"]["code"] == INVALID_PARAMS
    assert len(json.dumps(long_reason)) < 2_000
    unknown_method = harness.request("x" * 1_000_000)
    assert unknown_method["error"]["code"] == -32601
    assert len(json.dumps(unknown_method)) < 2_000  # the SDK's echo is bounded on the way out
    assert time.perf_counter() - started < 10


def test_malformed_id_values_never_yield_a_spoofed_id(harness: StdioServerHarness) -> None:
    harness.initialize()
    harness.send_raw('{"jsonrpc": "2.0", "id": {"id": "spoof"}, "method": 5}')
    assert harness.read_frame()["id"] is None
    harness.send_raw('{"jsonrpc": "2.0", "id": {"id": 7}, "method": "ping", "params": []}')
    assert harness.read_frame()["id"] is None
