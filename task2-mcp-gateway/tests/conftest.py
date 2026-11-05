"""Shared fixtures: an in-process gateway wired to the mock downstream.

Helpers are exposed as fixtures rather than imported, so the test modules
stay self-contained under pytest's importlib import mode.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette

from mcp_gateway.app import create_app
from mcp_gateway.config import GatewaySettings
from mock_mcp_server.app import MockState
from mock_mcp_server.app import create_app as create_mock_app

TOKENS: dict[str, dict[str, str]] = {
    "admin-token-0123456789abcdefghijklmnop": {"subject": "alice", "role": "admin"},
    "viewer-token-0123456789abcdefghijklmno": {"subject": "bob", "role": "viewer"},
}
HEADERS = {
    "admin": {"Authorization": "Bearer admin-token-0123456789abcdefghijklmnop"},
    "viewer": {"Authorization": "Bearer viewer-token-0123456789abcdefghijklmno"},
}
DOWNSTREAM_URL = "http://downstream/mcp"

GatewayFactory = Callable[..., Any]


def _rpc(
    request_id: Any, method: str, params: Any = None, *, notification: bool = False
) -> dict[str, Any]:
    frame: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if not notification:
        frame["id"] = request_id
    if params is not None:
        frame["params"] = params
    return frame


def _make_settings(**overrides: Any) -> GatewaySettings:
    base: dict[str, Any] = {
        "downstream_url": DOWNSTREAM_URL,
        "tokens_json": json.dumps({"tokens": TOKENS}),
    }
    base.update(overrides)
    return GatewaySettings(**base)


@pytest.fixture
def rpc() -> Callable[..., dict[str, Any]]:
    return _rpc


@pytest.fixture
def make_settings() -> Callable[..., GatewaySettings]:
    return _make_settings


@pytest.fixture
def tokens() -> dict[str, dict[str, str]]:
    return dict(TOKENS)


@pytest.fixture
def viewer() -> dict[str, str]:
    return dict(HEADERS["viewer"])


@pytest.fixture
def admin() -> dict[str, str]:
    return dict(HEADERS["admin"])


@pytest.fixture
def headers_for() -> Callable[[str], dict[str, str]]:
    return lambda role: dict(HEADERS[role])


@pytest.fixture
def mock_app() -> Starlette:
    return create_mock_app()


@pytest.fixture
def mock_state(mock_app: Starlette) -> MockState:
    state: MockState = mock_app.state.mock
    return state


@pytest.fixture
def gateway_for(mock_app: Starlette) -> GatewayFactory:
    """Async context manager factory: a client for a gateway built from ``settings``.

    The downstream defaults to the in-process mock; pass ``http_client`` to
    substitute a failing transport.
    """

    @asynccontextmanager
    async def factory(
        settings: GatewaySettings, http_client: httpx.AsyncClient | None = None
    ) -> AsyncIterator[httpx.AsyncClient]:
        downstream = http_client or httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_app), base_url="http://downstream"
        )
        app = create_app(settings, http_client=downstream)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway"
            ) as client:
                yield client
        finally:
            await downstream.aclose()

    return factory


@pytest.fixture
async def gateway(gateway_for: GatewayFactory) -> AsyncIterator[httpx.AsyncClient]:
    async with gateway_for(_make_settings()) as client:
        yield client
