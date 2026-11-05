"""Shared fixtures for the customer MCP server tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from customer_mcp_server.service import CustomerService, InMemoryCustomerRepository
from customer_mcp_server.testing import StdioServerHarness


@pytest.fixture
def service() -> CustomerService:
    return CustomerService(InMemoryCustomerRepository.with_sample_data())


@pytest.fixture
def harness() -> Iterator[StdioServerHarness]:
    server = StdioServerHarness()
    try:
        yield server
    finally:
        server.close()
