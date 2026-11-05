"""End-to-end behaviour of the gateway with the mock downstream (in-process)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from starlette.requests import ClientDisconnect

from mcp_gateway.app import RelayInterruptedError
from mcp_gateway.config import GatewaySettings
from mcp_gateway.jsonrpc import (
    DOWNSTREAM_TIMEOUT,
    DOWNSTREAM_UNAVAILABLE,
    GATEWAY_OVERLOADED,
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    UNAUTHENTICATED,
    UNAUTHORIZED_TOOL_CALL,
)
from mock_mcp_server.app import MockState

Rpc = Callable[..., dict[str, Any]]
Settings = Callable[..., GatewaySettings]
Factory = Callable[..., Any]


class _Body(httpx.AsyncByteStream):
    """A response body that is streamed (not pre-read), optionally failing at the end."""

    def __init__(self, *chunks: bytes, error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    async def __aiter__(self) -> Any:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


def _json_response(
    status: int = 200, body: Any = None, headers: list[tuple[bytes, bytes]] | None = None
) -> httpx.Response:
    payload = json.dumps(body or {"jsonrpc": "2.0", "id": 1, "result": {}}).encode()
    return httpx.Response(
        status,
        headers=[(b"content-type", b"application/json"), *(headers or [])],
        stream=_Body(payload),
    )


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def test_missing_token_is_rejected_with_challenge(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc
) -> None:
    response = await gateway.post("/mcp", json=rpc(1, "tools/list"))
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="mcp-gateway"'
    body = response.json()
    assert body["id"] is None
    assert body["error"]["code"] == UNAUTHENTICATED
    assert body["error"]["data"]["reason"] == "missing_token"
    assert mock_state.calls == []


@pytest.mark.parametrize(
    ("authorization", "reason"),
    [
        ("Bearer not-a-real-token", "invalid_token"),
        ("Bearer admin-token-12345", "invalid_token"),  # one character short
        ("Basic YWxpY2U6c2VjcmV0", "malformed_header"),
        ("Bearer", "malformed_header"),
    ],
)
async def test_bad_credentials_are_rejected(
    gateway: httpx.AsyncClient, rpc: Rpc, authorization: str, reason: str
) -> None:
    response = await gateway.post(
        "/mcp", json=rpc(1, "tools/list"), headers={"Authorization": authorization}
    )
    assert response.status_code == 401
    assert 'error="invalid_token"' in response.headers["www-authenticate"]
    assert response.json()["error"]["data"]["reason"] == reason


async def test_health_endpoint_needs_no_credentials(gateway: httpx.AsyncClient) -> None:
    response = await gateway.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Transparent forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["viewer", "admin"])
async def test_tools_list_is_forwarded_for_every_role(
    gateway: httpx.AsyncClient, rpc: Rpc, headers_for: Callable[[str], dict[str, str]], role: str
) -> None:
    response = await gateway.post("/mcp", json=rpc(2, "tools/list"), headers=headers_for(role))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-request-id"]
    tools = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert {"get_weather", "admin_reset_key"} <= tools


async def test_viewer_can_call_unrestricted_tool(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, viewer: dict[str, str]
) -> None:
    response = await gateway.post(
        "/mcp",
        json=rpc(3, "tools/call", {"name": "get_weather", "arguments": {"city": "Ankara"}}),
        headers=viewer,
    )
    assert response.status_code == 200
    assert response.json()["result"]["content"][0]["text"] == "Sunny, 22C in Ankara"
    assert [call.tool for call in mock_state.calls] == ["get_weather"]


async def test_initialize_and_unknown_methods_are_relayed_verbatim(
    gateway: httpx.AsyncClient, rpc: Rpc, viewer: dict[str, str]
) -> None:
    init = await gateway.post(
        "/mcp",
        json=rpc(1, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}}),
        headers=viewer,
    )
    assert init.json()["result"]["protocolVersion"] == "2025-03-26"

    unknown = await gateway.post("/mcp", json=rpc(2, "does/not/exist"), headers=viewer)
    assert unknown.status_code == 200
    assert unknown.json()["error"]["code"] == -32601  # produced downstream, relayed as-is


async def test_allowed_notification_is_relayed(
    gateway: httpx.AsyncClient, rpc: Rpc, viewer: dict[str, str]
) -> None:
    response = await gateway.post(
        "/mcp", json=rpc(None, "notifications/initialized", notification=True), headers=viewer
    )
    assert response.status_code == 202
    assert response.content == b""


async def test_get_and_delete_are_authenticated_and_relayed(
    gateway: httpx.AsyncClient, viewer: dict[str, str]
) -> None:
    anonymous = await gateway.get("/mcp")
    assert anonymous.status_code == 401
    # The mock only serves POST, so the downstream's 405 is relayed unchanged.
    relayed = await gateway.get("/mcp", headers=viewer)
    assert relayed.status_code == 405


# ---------------------------------------------------------------------------
# Tool-level authorisation
# ---------------------------------------------------------------------------


async def test_viewer_is_denied_admin_tool_without_reaching_downstream(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, viewer: dict[str, str]
) -> None:
    response = await gateway.post(
        "/mcp",
        json=rpc(4, "tools/call", {"name": "admin_reset_key", "arguments": {"user_id": "u1"}}),
        headers=viewer,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 4
    assert "result" not in body
    assert body["error"]["code"] == UNAUTHORIZED_TOOL_CALL
    assert body["error"]["message"] == "Unauthorized Tool Call"
    assert body["error"]["data"]["tool"] == "admin_reset_key"
    assert body["error"]["data"]["required_roles"] == ["admin"]
    assert body["error"]["data"]["role"] == "viewer"
    assert mock_state.calls == []


@pytest.mark.parametrize("name", ["Admin_Reset_Key", "ADMIN_DELETE_USER", " admin_reset_key"])
async def test_prefix_check_cannot_be_bypassed_with_casing_or_padding(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, viewer: dict[str, str], name: str
) -> None:
    response = await gateway.post("/mcp", json=rpc(5, "tools/call", {"name": name}), headers=viewer)
    assert response.json()["error"]["code"] == UNAUTHORIZED_TOOL_CALL
    assert mock_state.calls == []


async def test_admin_can_call_admin_tool(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, admin: dict[str, str]
) -> None:
    response = await gateway.post(
        "/mcp",
        json=rpc(6, "tools/call", {"name": "admin_reset_key", "arguments": {"user_id": "u1"}}),
        headers=admin,
    )
    assert response.status_code == 200
    assert response.json()["result"]["content"][0]["text"] == "API key rotated for user u1"
    assert mock_state.calls[0].identity["x-gateway-role"] == "admin"


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        (None, "params_not_object"),
        (["admin_reset_key"], "params_not_object"),
        ({}, "tool_name_missing"),
        ({"name": 5}, "tool_name_missing"),
        ({"name": ""}, "tool_name_missing"),
    ],
)
async def test_tool_calls_without_a_determinable_name_fail_closed(
    gateway: httpx.AsyncClient,
    mock_state: MockState,
    rpc: Rpc,
    admin: dict[str, str],
    params: Any,
    reason: str,
) -> None:
    response = await gateway.post("/mcp", json=rpc(7, "tools/call", params), headers=admin)
    assert response.status_code == 400
    body = response.json()
    assert body["id"] == 7
    assert body["error"]["code"] == INVALID_PARAMS
    assert body["error"]["data"]["reason"] == reason
    assert mock_state.calls == []


async def test_denied_notification_returns_403_with_null_id(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, viewer: dict[str, str]
) -> None:
    response = await gateway.post(
        "/mcp",
        json=rpc(None, "tools/call", {"name": "admin_reset_key"}, notification=True),
        headers=viewer,
    )
    assert response.status_code == 403
    body = response.json()
    assert body["id"] is None
    assert body["error"]["code"] == UNAUTHORIZED_TOOL_CALL
    assert mock_state.calls == []


@pytest.mark.parametrize("request_id", ["req-abc", 0, 1.5, None])
async def test_request_ids_are_preserved_in_gateway_errors(
    gateway: httpx.AsyncClient, rpc: Rpc, viewer: dict[str, str], request_id: Any
) -> None:
    response = await gateway.post(
        "/mcp", json=rpc(request_id, "tools/call", {"name": "admin_x"}), headers=viewer
    )
    assert response.json()["id"] == request_id


async def test_denied_status_can_be_configured_to_403(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    async with gateway_for(make_settings(denied_http_status=403)) as client:
        response = await client.post(
            "/mcp", json=rpc(1, "tools/call", {"name": "admin_x"}), headers=viewer
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == UNAUTHORIZED_TOOL_CALL


async def test_custom_prefixes_and_roles(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    tokens = {
        "tokens": {
            "sre-token-0123456789abcdefghijklmnopqr": {"subject": "dan", "role": "sre"},
            "viewer-token-0123456789abcdefghijklmno": {"subject": "bob", "role": "viewer"},
        }
    }
    settings = make_settings(
        tokens_json=json.dumps(tokens),
        admin_tool_prefixes=("admin_", "danger_"),
        admin_roles=("admin", "sre"),
    )
    sre = {"Authorization": "Bearer sre-token-0123456789abcdefghijklmnopqr"}
    async with gateway_for(settings) as client:
        allowed = await client.post(
            "/mcp", json=rpc(1, "tools/call", {"name": "admin_reset_key"}), headers=sre
        )
        denied = await client.post(
            "/mcp", json=rpc(2, "tools/call", {"name": "danger_wipe"}), headers=viewer
        )
    assert "result" in allowed.json()
    assert denied.json()["error"]["code"] == UNAUTHORIZED_TOOL_CALL


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


async def test_fully_authorised_batch_is_forwarded(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, viewer: dict[str, str]
) -> None:
    batch = [
        rpc(10, "ping"),
        rpc(11, "tools/call", {"name": "echo", "arguments": {"message": "hi"}}),
        rpc(None, "notifications/progress", notification=True),
    ]
    response = await gateway.post("/mcp", json=batch, headers=viewer)
    assert response.status_code == 200
    results = {item["id"]: item for item in response.json()}
    assert set(results) == {10, 11}
    assert results[11]["result"]["content"][0]["text"] == "hi"
    assert [call.tool for call in mock_state.calls] == ["echo"]


async def test_batch_with_one_denied_call_is_rejected_as_a_whole(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, viewer: dict[str, str]
) -> None:
    batch = [
        rpc(12, "ping"),
        rpc(13, "tools/call", {"name": "admin_delete_user", "arguments": {"user_id": "u"}}),
        rpc(None, "notifications/progress", notification=True),
    ]
    response = await gateway.post("/mcp", json=batch, headers=viewer)
    assert response.status_code == 200
    errors = {item["id"]: item["error"] for item in response.json()}
    assert set(errors) == {12, 13}  # the notification gets no reply
    assert errors[13]["code"] == UNAUTHORIZED_TOOL_CALL
    assert errors[13]["data"]["reason"] == "role_not_permitted"
    assert errors[12]["code"] == UNAUTHORIZED_TOOL_CALL
    assert errors[12]["data"]["reason"] == "batch_contains_unauthorized_tool_call"
    assert mock_state.calls == []


# ---------------------------------------------------------------------------
# Malformed payloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "code", "expected_id"),
    [
        (b"{not json", PARSE_ERROR, None),
        (b"", PARSE_ERROR, None),
        (b"[]", INVALID_REQUEST, None),
        (b"42", INVALID_REQUEST, None),
        (b'{"jsonrpc": "1.0", "id": 9, "method": "ping"}', INVALID_REQUEST, 9),
        (b'{"jsonrpc": "2.0", "id": 9}', INVALID_REQUEST, 9),
        (b'{"jsonrpc": "2.0", "id": 9, "method": "x", "params": "p"}', INVALID_REQUEST, 9),
        (b'[{"jsonrpc": "2.0", "id": 1, "method": "ping"}, 5]', INVALID_REQUEST, None),
        (b"[" * 100_000, PARSE_ERROR, None),  # recursion bomb
    ],
)
async def test_malformed_payloads_get_standard_errors(
    gateway: httpx.AsyncClient,
    mock_state: MockState,
    viewer: dict[str, str],
    content: bytes,
    code: int,
    expected_id: Any,
) -> None:
    response = await gateway.post(
        "/mcp", content=content, headers={**viewer, "Content-Type": "application/json"}
    )
    assert response.status_code == 400
    body = response.json()
    assert body["id"] == expected_id
    assert body["error"]["code"] == code
    assert mock_state.calls == []


async def test_oversized_body_is_rejected_before_parsing(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    async with gateway_for(make_settings(max_body_bytes=64)) as client:
        big = rpc(1, "tools/call", {"name": "echo", "arguments": {"message": "x" * 500}})
        response = await client.post("/mcp", json=big, headers=viewer)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == INVALID_REQUEST


# ---------------------------------------------------------------------------
# Header hygiene and identity propagation
# ---------------------------------------------------------------------------


async def test_identity_headers_are_injected_and_client_values_cannot_spoof_them(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, viewer: dict[str, str]
) -> None:
    response = await gateway.post(
        "/mcp",
        json=rpc(1, "tools/call", {"name": "whoami"}),
        headers={
            **viewer,
            "X-Gateway-Role": "admin",
            "X-Gateway-Subject": "mallory",
            "X-Request-Id": "trace-42",
            "X-Forwarded-For": "203.0.113.9",
        },
    )
    seen = response.json()["result"]["structuredContent"]
    assert seen["x-gateway-subject"] == "bob"
    assert seen["x-gateway-role"] == "viewer"
    assert seen["x-request-id"] == "trace-42"
    assert seen["x-forwarded-for"].startswith("203.0.113.9, ")
    assert seen["authorization_present"] is False
    assert response.headers["x-request-id"] == "trace-42"
    assert mock_state.calls[0].authorization_present is False


async def test_malformed_request_ids_are_replaced(
    gateway: httpx.AsyncClient, rpc: Rpc, viewer: dict[str, str]
) -> None:
    response = await gateway.post(
        "/mcp",
        json=rpc(1, "tools/call", {"name": "whoami"}),
        headers={**viewer, "X-Request-Id": "bad id with spaces"},
    )
    assigned = response.headers["x-request-id"]
    assert assigned != "bad id with spaces"
    assert response.json()["result"]["structuredContent"]["x-request-id"] == assigned


async def test_service_token_replaces_client_credential(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    async with gateway_for(make_settings(downstream_service_token="svc-secret")) as client:
        response = await client.post(
            "/mcp", json=rpc(1, "tools/call", {"name": "whoami"}), headers=viewer
        )
    assert response.json()["result"]["structuredContent"]["authorization_present"] is True


# ---------------------------------------------------------------------------
# Downstream failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (httpx.ConnectError("connection refused"), 502, DOWNSTREAM_UNAVAILABLE),
        (httpx.RemoteProtocolError("peer closed"), 502, DOWNSTREAM_UNAVAILABLE),
        (httpx.ConnectTimeout("slow connect"), 504, DOWNSTREAM_TIMEOUT),
        (httpx.ReadTimeout("slow read"), 504, DOWNSTREAM_TIMEOUT),
    ],
)
async def test_downstream_failures_are_sanitised(
    gateway_for: Factory,
    make_settings: Settings,
    rpc: Rpc,
    viewer: dict[str, str],
    exc: Exception,
    status: int,
    code: int,
) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        raise exc

    failing_client = httpx.AsyncClient(transport=httpx.MockTransport(failing))
    async with gateway_for(make_settings(), failing_client) as client:
        single = await client.post("/mcp", json=rpc("r1", "ping"), headers=viewer)
        batch = await client.post(
            "/mcp",
            json=[rpc(1, "ping"), rpc(None, "notifications/x", notification=True)],
            headers=viewer,
        )
    assert single.status_code == status
    body = single.json()
    assert body["id"] == "r1"
    assert body["error"]["code"] == code
    assert "connection refused" not in single.text
    assert "slow" not in single.text
    assert batch.status_code == status
    assert [item["id"] for item in batch.json()] == [1]


# ---------------------------------------------------------------------------
# Long-lived streams
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream_timeout", [123.0, None])
async def test_get_streams_get_their_own_idle_timeout(
    gateway_for: Factory,
    make_settings: Settings,
    rpc: Rpc,
    viewer: dict[str, str],
    stream_timeout: float | None,
) -> None:
    seen: dict[str, dict[str, float | None]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.method] = dict(request.extensions["timeout"])
        return _json_response()

    downstream = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=7.0)
    settings = make_settings(stream_read_timeout_seconds=stream_timeout)
    async with gateway_for(settings, downstream) as client:
        await client.post("/mcp", json=rpc(1, "ping"), headers=viewer)
        await client.get("/mcp", headers=viewer)
    # POST keeps the client's request/response timeout; GET uses the idle bound.
    assert seen["POST"]["read"] == 7.0
    assert seen["GET"]["read"] == stream_timeout
    assert seen["GET"]["connect"] == settings.connect_timeout_seconds


async def test_mid_stream_downstream_failure_is_logged_and_not_hidden(
    gateway_for: Factory,
    make_settings: Settings,
    rpc: Rpc,
    viewer: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _Body(
            b"event: message\ndata: {}\n\n", error=httpx.ReadTimeout("downstream went silent")
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    downstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), downstream) as client:
        # The failure surfaces as a broken connection, never as a clean end of stream.
        with caplog.at_level(logging.WARNING), pytest.raises(RelayInterruptedError):
            await client.get("/mcp", headers={**viewer, "X-Request-Id": "stream-7"})
    records = [r for r in caplog.records if getattr(r, "event", "") == "proxy.stream_interrupted"]
    assert len(records) == 1
    assert records[0].request_id == "stream-7"  # type: ignore[attr-defined]
    assert records[0].kind == "timeout"  # type: ignore[attr-defined]
    # Logged exactly once: the generic handler stays quiet for relay interruptions.
    assert [r for r in caplog.records if r.getMessage() == "unhandled error"] == []


# ---------------------------------------------------------------------------
# Round two: capacity protection, parser strictness, robustness
# ---------------------------------------------------------------------------


async def test_per_subject_concurrency_cap(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    async with gateway_for(make_settings(max_inflight_per_subject=2)) as client:
        slow = {**viewer, "X-Mock-Delay-Ms": "300"}

        async def ping() -> httpx.Response:
            return await client.post("/mcp", json=rpc(1, "ping"), headers=slow)

        first, second = asyncio.ensure_future(ping()), asyncio.ensure_future(ping())
        await asyncio.sleep(0.05)
        third = await client.post("/mcp", json=rpc(3, "ping"), headers=viewer)
        responses = await asyncio.gather(first, second)
        after = await client.post("/mcp", json=rpc(4, "ping"), headers=viewer)
    assert [r.status_code for r in responses] == [200, 200]
    assert third.status_code == 429
    assert third.headers["retry-after"] == "1"
    assert third.json()["error"]["code"] == GATEWAY_OVERLOADED
    assert after.status_code == 200  # slots are released when the responses complete


async def test_pool_exhaustion_is_reported_as_gateway_capacity(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.PoolTimeout("no connection available")

    downstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), downstream) as client:
        response = await client.post("/mcp", json=rpc("p", "ping"), headers=viewer)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert response.json()["error"]["code"] == GATEWAY_OVERLOADED
    assert response.json()["id"] == "p"


async def test_duplicate_members_are_rejected_before_policy(
    gateway: httpx.AsyncClient, mock_state: MockState, viewer: dict[str, str]
) -> None:
    first_wins = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"admin_reset_key","name":"get_weather"}}'
    )
    method_twice = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"admin_reset_key"},"method":"ping"}'
    )
    for body in (first_wins, method_twice):
        response = await gateway.post(
            "/mcp", content=body, headers={**viewer, "Content-Type": "application/json"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == INVALID_REQUEST
        assert "duplicate" in response.json()["error"]["message"]
    assert mock_state.calls == []


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (b'{"jsonrpc":"2.0","id":1e400,"method":"ping"}', 400, INVALID_REQUEST),
        (b'{"jsonrpc":"2.0","id":NaN,"method":"ping"}', 400, PARSE_ERROR),
        (b'{"jsonrpc":"2.0","id":Infinity,"method":"ping"}', 400, PARSE_ERROR),
        (b'{"jsonrpc":"2.0","id":' + b"9" * 5000 + b',"method":"ping"}', 400, PARSE_ERROR),
        (
            b'{"jsonrpc":"2.0","id":"\\udc00","method":"tools/call","params":{}}',
            400,
            INVALID_PARAMS,
        ),
        (
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_x\\udc00"}}',
            400,
            INVALID_PARAMS,
        ),
        (
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call\\u0007","params":{"name":"x"}}',
            400,
            INVALID_REQUEST,
        ),
    ],
)
async def test_hostile_ids_and_names_never_cause_a_500(
    gateway: httpx.AsyncClient,
    mock_state: MockState,
    viewer: dict[str, str],
    body: bytes,
    status: int,
    code: int,
) -> None:
    response = await gateway.post(
        "/mcp", content=body, headers={**viewer, "Content-Type": "application/json"}
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert mock_state.calls == []


async def test_non_latin1_downstream_headers_are_relayed_as_received(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(headers=[(b"x-note", "price \u20ac".encode())])

    downstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), downstream) as client:
        response = await client.post("/mcp", json=rpc(1, "ping"), headers=viewer)
    assert response.status_code == 200
    assert response.headers["x-note"] == "price \u20ac"


async def test_batch_rejected_for_an_invalid_element_labels_siblings_invalid(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, admin: dict[str, str]
) -> None:
    batch = [rpc(1, "ping"), rpc(2, "tools/call", {})]
    response = await gateway.post("/mcp", json=batch, headers=admin)
    assert response.status_code == 400
    errors = {item["id"]: item["error"] for item in response.json()}
    assert errors[2]["code"] == INVALID_PARAMS
    assert errors[1]["code"] == INVALID_REQUEST
    assert errors[1]["data"]["reason"] == "batch_contains_invalid_request"
    assert mock_state.calls == []


async def test_batch_size_cap(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    async with gateway_for(make_settings(max_batch_size=3)) as client:
        ok = await client.post("/mcp", json=[rpc(i, "ping") for i in range(3)], headers=viewer)
        too_many = await client.post(
            "/mcp", json=[rpc(i, "ping") for i in range(4)], headers=viewer
        )
    assert ok.status_code == 200
    assert too_many.status_code == 400
    assert too_many.json()["error"]["code"] == INVALID_REQUEST
    assert "exceeds 3" in too_many.json()["error"]["message"]


async def test_routing_errors_are_json_rpc_errors_and_nothing_redirects(
    gateway: httpx.AsyncClient, rpc: Rpc, viewer: dict[str, str]
) -> None:
    missing = await gateway.get("/nope", headers=viewer)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == INVALID_REQUEST
    wrong_method = await gateway.put("/mcp", headers=viewer)
    assert wrong_method.status_code == 405
    assert "POST" in wrong_method.headers["allow"]
    assert wrong_method.json()["error"]["code"] == INVALID_REQUEST
    trailing = await gateway.post(
        "/mcp/", json=rpc(1, "ping"), headers={**viewer, "Host": "evil.example"}
    )
    assert trailing.status_code == 404  # never a 307 built from the client's Host header


async def test_downstream_query_strings_are_merged(
    gateway_for: Factory, make_settings: Settings, viewer: dict[str, str]
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return _json_response()

    downstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = make_settings(downstream_url="http://downstream/mcp?tenant=a")
    async with gateway_for(settings, downstream) as client:
        await client.get("/mcp?session=1", headers=viewer)
        await client.get("/mcp", headers=viewer)
    assert seen == ["http://downstream/mcp?tenant=a&session=1", "http://downstream/mcp?tenant=a"]


async def test_client_disconnect_during_upload_is_a_warning_not_an_error(
    gateway: httpx.AsyncClient,
    rpc: Rpc,
    viewer: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def gone(request: Any, limit: int) -> bytes:
        raise ClientDisconnect

    monkeypatch.setattr("mcp_gateway.app.read_body_limited", gone)
    with caplog.at_level(logging.WARNING):
        response = await gateway.post("/mcp", json=rpc(1, "ping"), headers=viewer)
    assert response.status_code == 400
    events = [getattr(r, "event", "") for r in caplog.records]
    assert "client.disconnected" in events
    assert all(r.levelno < logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# Security review: encodings, bounds, raw header relay, slot release, liveness
# ---------------------------------------------------------------------------


async def test_non_utf8_bodies_are_parse_errors(
    gateway: httpx.AsyncClient, mock_state: MockState, viewer: dict[str, str]
) -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode("utf-16")
    response = await gateway.post(
        "/mcp", content=body, headers={**viewer, "Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == PARSE_ERROR
    assert "UTF-8" in response.json()["error"]["message"]
    assert mock_state.calls == []


async def test_padded_tool_names_are_refused_with_a_small_response(
    gateway: httpx.AsyncClient, mock_state: MockState, rpc: Rpc, viewer: dict[str, str]
) -> None:
    padded = " " * 300_000 + "admin_reset_key"
    response = await gateway.post(
        "/mcp", json=rpc(1, "tools/call", {"name": padded}), headers=viewer
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == INVALID_PARAMS
    assert len(response.content) < 2048
    assert mock_state.calls == []


async def test_client_headers_are_relayed_as_raw_bytes(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    seen: list[tuple[bytes, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.extend(request.headers.raw)
        return _json_response()

    downstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), downstream) as client:
        response = await client.post(
            "/mcp",
            json=rpc(1, "ping"),
            headers=httpx.Headers([(b"x-note", "price \u20ac".encode()), *viewer.items()]),
        )
    assert response.status_code == 200
    assert (b"x-note", "price \u20ac".encode()) in seen
    names = [name for name, _ in seen]
    assert b"x-gateway-subject" in names
    assert b"authorization" not in names


async def test_downstream_server_and_date_headers_are_not_relayed(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            headers=[
                (b"server", b"SecretServer/9"),
                (b"date", b"Mon, 01 Jan 2024 00:00:00 GMT"),
                (b"x-kept", b"yes"),
            ]
        )

    downstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), downstream) as client:
        response = await client.post("/mcp", json=rpc(1, "ping"), headers=viewer)
    assert response.headers.get("server") != "SecretServer/9"
    assert "date" not in response.headers or "2024" not in response.headers["date"]
    assert response.headers["x-kept"] == "yes"


async def test_health_discloses_nothing(gateway: httpx.AsyncClient) -> None:
    response = await gateway.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store"


def test_request_ids_with_trailing_newlines_are_replaced() -> None:
    from mcp_gateway.app import resolve_request_id

    assert resolve_request_id("trace-1\n") != "trace-1\n"
    assert resolve_request_id("trace-1") == "trace-1"


async def test_relayed_response_cleans_up_when_the_response_cannot_start() -> None:
    from mcp_gateway.app import RelayedResponse

    closed = []

    async def on_close() -> None:
        closed.append(True)

    async def body() -> Any:
        yield b"data"

    async def failing_send(message: Any) -> None:
        raise RuntimeError("client went away before the status line")

    async def receive() -> Any:
        await asyncio.sleep(3600)  # no disconnect ever arrives; the failing send ends it
        return {"type": "http.disconnect"}

    response = RelayedResponse(body(), status_code=200, on_close=on_close)
    with pytest.raises(RuntimeError):
        await response({"type": "http"}, receive, failing_send)
    assert closed == [True]


async def test_slot_is_released_when_a_relayed_stream_fails_to_start(
    gateway_for: Factory, make_settings: Settings, rpc: Rpc, viewer: dict[str, str]
) -> None:
    """End to end: a downstream whose stream dies immediately leaves no slot taken."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[(b"content-type", b"application/json")],
            stream=_Body(error=httpx.ReadError("gone")),
        )

    downstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(max_inflight_per_subject=1), downstream) as client:
        for _ in range(3):
            # The transport re-raises the relay failure in-process; a real server would
            # just drop the connection. Either way the slot must come back.
            with contextlib.suppress(Exception):
                await client.post("/mcp", json=rpc(1, "ping"), headers=viewer)
        service = client._transport.app.state.service  # type: ignore[attr-defined]
        assert service.inflight.count("bob") == 0
