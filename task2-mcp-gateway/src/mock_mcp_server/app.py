"""A deliberately small MCP-shaped JSON-RPC server used as the gateway's downstream.

It implements ``initialize``, ``ping``, ``tools/list`` and ``tools/call`` with
plain JSON responses, understands batches and notifications, records every
tool call it executes (so tests and demos can prove what did or did not reach
it) and exposes a ``whoami`` tool that reflects the identity headers it
received from the gateway.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "mock-downstream-mcp", "version": "1.0.0"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Weather for a city (static mock data).",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "echo",
        "description": "Echo a message back.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    {
        "name": "whoami",
        "description": "Report the identity headers the downstream received.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "admin_reset_key",
        "description": "Rotate the API key of a user (administrative).",
        "inputSchema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "admin_delete_user",
        "description": "Delete a user account (administrative).",
        "inputSchema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
]

_IDENTITY_HEADERS = ("x-gateway-subject", "x-gateway-role", "x-request-id", "x-forwarded-for")


@dataclass
class CallRecord:
    tool: str
    arguments: dict[str, Any]
    identity: dict[str, str | None]
    authorization_present: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "identity": self.identity,
            "authorization_present": self.authorization_present,
        }


@dataclass
class MockState:
    calls: list[CallRecord] = field(default_factory=list)


def _result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["isError"] = True
    return result


def _execute_tool(name: str, arguments: dict[str, Any], request: Request, state: MockState) -> Any:
    identity = {h: request.headers.get(h) for h in _IDENTITY_HEADERS}
    state.calls.append(
        CallRecord(
            tool=name,
            arguments=arguments,
            identity=identity,
            authorization_present="authorization" in request.headers,
        )
    )
    if name == "get_weather":
        city = arguments.get("city")
        if not isinstance(city, str):
            return _text_result("city is required", is_error=True)
        return _text_result(f"Sunny, 22C in {city}")
    if name == "echo":
        return _text_result(str(arguments.get("message", "")))
    if name == "whoami":
        payload = {**identity, "authorization_present": "authorization" in request.headers}
        result = _text_result(json.dumps(payload))
        result["structuredContent"] = payload
        return result
    if name == "admin_reset_key":
        return _text_result(f"API key rotated for user {arguments.get('user_id')}")
    if name == "admin_delete_user":
        return _text_result(f"Deleted user {arguments.get('user_id')}")
    return None


def _dispatch(message: dict[str, Any], request: Request, state: MockState) -> dict[str, Any] | None:
    request_id = message.get("id")
    is_notification = "id" not in message
    method = message.get("method")
    params = message.get("params") or {}

    if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
        return _error(request_id, -32600, "Invalid Request")
    if is_notification:
        return None
    if method == "initialize":
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        return _result(
            request_id,
            {
                "protocolVersion": requested or PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(request_id, {"tools": TOOLS})
    if method == "tools/call":
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            return _error(request_id, -32602, "Invalid params")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(request_id, -32602, "Invalid params: arguments must be an object")
        outcome = _execute_tool(params["name"], arguments, request, state)
        if outcome is None:
            return _error(request_id, -32602, f"Unknown tool: {params['name']}")
        return _result(request_id, outcome)
    return _error(request_id, -32601, "Method not found")


def create_app() -> Starlette:
    state = MockState()

    async def mcp(request: Request) -> Response:
        delay_ms = request.headers.get("x-mock-delay-ms")
        if delay_ms and delay_ms.isdigit():
            await asyncio.sleep(int(delay_ms) / 1000)
        try:
            payload = json.loads(await request.body())
        except json.JSONDecodeError:
            return JSONResponse(_error(None, -32700, "Parse error"), status_code=400)

        if isinstance(payload, list):
            responses = [
                r for r in (_dispatch(m, request, state) for m in payload) if r is not None
            ]
            if not responses:
                return Response(status_code=202)
            return JSONResponse(responses)
        if not isinstance(payload, dict):
            return JSONResponse(_error(None, -32600, "Invalid Request"), status_code=400)
        response = _dispatch(payload, request, state)
        if response is None:
            return Response(status_code=202)
        return JSONResponse(response)

    async def calls(request: Request) -> Response:
        if request.method == "DELETE":
            state.calls.clear()
            return Response(status_code=204)
        return JSONResponse([c.as_dict() for c in state.calls])

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", "service": "mock-downstream-mcp"})

    app = Starlette(
        routes=[
            Route("/mcp", mcp, methods=["POST"]),
            Route("/__mock/calls", calls, methods=["GET", "DELETE"]),
            Route("/healthz", health, methods=["GET"]),
        ]
    )
    app.state.mock = state
    return app
