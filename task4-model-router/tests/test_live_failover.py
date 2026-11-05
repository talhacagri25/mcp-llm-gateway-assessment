"""Failover over real sockets with the specified 3000 ms deadline."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from mock_model_provider.app import create_app as create_provider_app
from model_router.app import create_app
from model_router.config import ProviderSettings, RouterSettings, TenantSettings
from model_router.testing import UvicornTestServer

pytestmark = pytest.mark.integration

KEY = "live-tenant-key-0123456789abcdefghij"
HEADERS = {"Authorization": f"Bearer {KEY}"}
BODY = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}


@pytest.fixture(scope="module")
def router_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    db: Path = tmp_path_factory.mktemp("ledger") / "router.sqlite3"
    with (
        UvicornTestServer(create_provider_app("primary", "slow", 4000)) as primary,
        UvicornTestServer(create_provider_app("secondary", "ok", 0)) as secondary,
    ):
        settings = RouterSettings(
            primary=ProviderSettings("primary", primary.base_url, timeout_ms=3000),
            secondary=ProviderSettings("secondary", secondary.base_url, timeout_ms=3000),
            tenants={KEY: TenantSettings("live")},
            database_path=db,
        )
        with UvicornTestServer(create_app(settings)) as router:
            yield router.base_url


def test_primary_timeout_after_3000ms_fails_over_to_secondary(router_url: str) -> None:
    started = time.perf_counter()
    response = httpx.post(
        f"{router_url}/v1/chat/completions", json=BODY, headers=HEADERS, timeout=30
    )
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert response.headers["x-model-router-provider"] == "secondary"
    # Waited for the deadline, not for the 4 s provider.
    assert 3.0 <= elapsed < 3.9


def test_primary_429_fails_over_immediately(router_url: str) -> None:
    body = {**BODY, "mock": {"primary": {"behavior": "rate_limited"}}}
    started = time.perf_counter()
    response = httpx.post(
        f"{router_url}/v1/chat/completions", json=body, headers=HEADERS, timeout=30
    )
    assert response.status_code == 200
    assert response.headers["x-model-router-provider"] == "secondary"
    assert time.perf_counter() - started < 1.0
