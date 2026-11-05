"""Interoperability with the official MCP Python client.

Two flavours: an in-process connection (fast, exercises the handlers through
the SDK's dispatcher) and a subprocess connection over stdio (exercises the
real transport the way an agent host would).
"""

from __future__ import annotations

import sys

import pytest
from mcp import Client, MCPError, StdioServerParameters

from customer_mcp_server.server import build_server
from customer_mcp_server.service import CustomerService
from customer_mcp_server.testing import subprocess_environment

INVALID_PARAMS = -32602


async def test_in_process_client_round_trip(service: CustomerService) -> None:
    async with Client(build_server(service)) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {"get_customer_record", "trigger_refund"}

        result = await client.call_tool("get_customer_record", {"customer_id": "CUST-00005"})
        assert result.is_error is False
        assert result.structured_content["currency"] == "EUR"

        with pytest.raises(MCPError) as exc_info:
            await client.call_tool("trigger_refund", {"customer_id": "CUST-00005", "amount": -1})
        assert exc_info.value.code == INVALID_PARAMS
        paths = {issue["path"] for issue in exc_info.value.data["issues"]}
        assert paths == {"amount", "reason"}

        with pytest.raises(MCPError) as exc_info:
            await client.call_tool("no_such_tool", {})
        assert exc_info.value.code == INVALID_PARAMS

        business = await client.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-00004", "amount": 1, "reason": "Closed account refund"},
        )
        assert business.is_error is True
        assert business.structured_content["error"]["code"] == "customer_not_eligible"


@pytest.mark.integration
async def test_stdio_client_round_trip() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "customer_mcp_server", "--log-level", "DEBUG"],
        env=subprocess_environment(),
    )
    async with Client(params) as client:
        tools = await client.list_tools()
        assert len(tools.tools) == 2

        receipt = await client.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-00001", "amount": 10, "reason": "Goodwill gesture for delay"},
        )
        assert receipt.is_error is False
        assert receipt.structured_content["remaining_balance"] == 240.0

        with pytest.raises(MCPError) as exc_info:
            await client.call_tool("get_customer_record", {"customer_id": "CUST-1"})
        assert exc_info.value.code == INVALID_PARAMS
