"""Fixtures: in-process router wired to two scriptable mock providers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette

from mock_model_provider.app import ProviderState
from mock_model_provider.app import create_app as create_provider_app
from model_router.app import create_app
from model_router.config import ProviderSettings, RouterSettings, TenantSettings
from model_router.storage import TokenLedgerStore

TENANT_A_KEY = "tenant-a-key-0123456789abcdefghijklmnop"
TENANT_B_KEY = "tenant-b-key-0123456789abcdefghijklmnop"
TENANT_B_LIMIT = 2000


class HostRoutingTransport(httpx.AsyncBaseTransport):
    """Dispatch requests to in-process ASGI apps by host name."""

    def __init__(self, apps: dict[str, Starlette]) -> None:
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self._transports.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError(f"no route to host {request.url.host}")
        return await transport.handle_async_request(request)

    async def aclose(self) -> None:
        for transport in self._transports.values():
            await transport.aclose()


class FakeClock:
    """Deterministic time source shared by the limiter and the circuit breakers."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger" / "model_router.sqlite3"


@pytest.fixture
def store(db_path: Path) -> Iterator[TokenLedgerStore]:
    ledger = TokenLedgerStore(db_path)
    try:
        yield ledger
    finally:
        ledger.close()


@pytest.fixture
def providers() -> dict[str, Starlette]:
    return {
        "primary": create_provider_app("primary", "ok", 5000),
        "secondary": create_provider_app("secondary", "ok", 5000),
    }


@pytest.fixture
def provider_state(providers: dict[str, Starlette]) -> Callable[[str], ProviderState]:
    def lookup(name: str) -> ProviderState:
        state: ProviderState = providers[name].state.provider.state
        return state

    return lookup


@pytest.fixture
def make_settings(db_path: Path) -> Callable[..., RouterSettings]:
    def factory(**overrides: Any) -> RouterSettings:
        base: dict[str, Any] = {
            "primary": ProviderSettings(
                "primary",
                "http://primary",
                api_key="primary-key",
                model="primary-model",
                timeout_ms=300,
            ),
            "secondary": ProviderSettings("secondary", "http://secondary", timeout_ms=300),
            "tenants": {
                TENANT_A_KEY: TenantSettings("tenant-a"),
                TENANT_B_KEY: TenantSettings("tenant-b", tokens_per_minute=TENANT_B_LIMIT),
            },
            "database_path": db_path,
            "tokens_per_minute": 50_000,
            "circuit_failure_threshold": 3,
            "circuit_recovery_seconds": 30.0,
            "forward_headers": ("traceparent", "tracestate", "x-mock-controls"),
        }
        base.update(overrides)
        return RouterSettings(**base)

    return factory


@pytest.fixture
async def http_client(providers: dict[str, Starlette]) -> AsyncIterator[httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=HostRoutingTransport(providers))
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def app_for(
    http_client: httpx.AsyncClient, store: TokenLedgerStore, clock: FakeClock
) -> Callable[..., Any]:
    @asynccontextmanager
    async def factory(settings: RouterSettings) -> AsyncIterator[httpx.AsyncClient]:
        app = create_app(
            settings, http_client=http_client, store=store, clock=clock, monotonic_clock=clock
        )
        # raise_app_exceptions=False mirrors a real server: the error handler's
        # response is what the client sees, while the traceback goes to the log.
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://router", timeout=30
        ) as client:
            yield client

    return factory


@pytest.fixture
async def client(
    app_for: Callable[..., Any], make_settings: Callable[..., RouterSettings]
) -> AsyncIterator[httpx.AsyncClient]:
    async with app_for(make_settings()) as router_client:
        yield router_client


@pytest.fixture
def tenant_a() -> dict[str, str]:
    return {"Authorization": f"Bearer {TENANT_A_KEY}"}


@pytest.fixture
def tenant_b() -> dict[str, str]:
    return {"Authorization": f"Bearer {TENANT_B_KEY}"}


@pytest.fixture
def completion() -> Callable[..., dict[str, Any]]:
    def build(max_tokens: int | None = 50, **mock: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": "client-model",
            "messages": [{"role": "user", "content": "Say hello in five words."}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if mock:
            body["mock"] = mock
        return body

    return build
