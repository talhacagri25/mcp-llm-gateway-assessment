"""The gateway in front of a real MCP server, driven by the official MCP client.

Both ends listen on real sockets (uvicorn in background threads), so this
covers the Streamable HTTP transport exactly the way an agent host uses it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
import mcp_types as types
import pytest
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server

from mcp_gateway.app import create_app
from mcp_gateway.config import GatewaySettings
from mcp_gateway.testing import UvicornTestServer

pytestmark = pytest.mark.integration

UNAUTHORIZED_TOOL_CALL = -32001


def build_downstream(calls: list[str]) -> Server[Any]:
    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        schema = {"type": "object", "properties": {"value": {"type": "string"}}}
        return types.ListToolsResult(
            tools=[
                types.Tool(name="echo", input_schema=schema),
                types.Tool(name="admin_reset_key", input_schema=schema),
            ]
        )

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        calls.append(params.name)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"{params.name} executed")]
        )

    return Server(
        "real-downstream", version="1.0.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool
    )


@pytest.fixture
def stack(tokens: dict[str, dict[str, str]]) -> Iterator[tuple[str, list[str]]]:
    calls: list[str] = []
    downstream_app = build_downstream(calls).streamable_http_app(
        json_response=True, stateless_http=True
    )
    with UvicornTestServer(downstream_app) as downstream:
        settings = GatewaySettings(
            downstream_url=f"{downstream.base_url}/mcp",
            tokens_json=json.dumps({"tokens": tokens}),
        )
        with UvicornTestServer(create_app(settings)) as gateway:
            yield f"{gateway.base_url}/mcp", calls


@asynccontextmanager
async def mcp_client(url: str, token: str | None) -> AsyncIterator[Client]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    http = httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30.0))
    try:
        async with Client(streamable_http_client(url, http_client=http)) as client:
            yield client
    finally:
        await http.aclose()


async def test_viewer_session_through_the_gateway(stack: tuple[str, list[str]]) -> None:
    url, calls = stack
    async with mcp_client(url, "viewer-token-0123456789abcdefghijklmno") as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {"echo", "admin_reset_key"}

        allowed = await client.call_tool("echo", {"value": "hi"})
        assert allowed.is_error is False
        assert allowed.content[0].type == "text"

        with pytest.raises(MCPError) as exc_info:
            await client.call_tool("admin_reset_key", {"value": "u1"})
        assert exc_info.value.code == UNAUTHORIZED_TOOL_CALL
        assert exc_info.value.data["tool"] == "admin_reset_key"
    assert calls == ["echo"]


async def test_admin_session_through_the_gateway(stack: tuple[str, list[str]]) -> None:
    url, calls = stack
    async with mcp_client(url, "admin-token-0123456789abcdefghijklmnop") as client:
        result = await client.call_tool("admin_reset_key", {"value": "u1"})
        assert result.is_error is False
    assert calls == ["admin_reset_key"]


async def test_anonymous_client_cannot_establish_a_session(
    stack: tuple[str, list[str]],
) -> None:
    url, calls = stack
    with pytest.raises(Exception):  # noqa: B017, PT011 - transport error type is SDK internal
        async with mcp_client(url, None) as client:
            await client.list_tools()
    assert calls == []
