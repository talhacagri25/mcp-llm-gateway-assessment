"""HTTP layer: authentication, validation, standardised errors and headers."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from model_router.__main__ import main
from model_router.config import ProviderSettings, RouterSettings, TenantSettings
from model_router.router import ModelRouter
from model_router.storage import TokenLedgerStore

from .conftest import TENANT_A_KEY, TENANT_B_KEY, TENANT_B_LIMIT

Body = Callable[..., dict[str, Any]]
Settings = Callable[..., RouterSettings]
INTERNAL_MARKERS = ("Traceback", "http://primary", "http://secondary", "internal detail", 'File "')


async def test_missing_or_invalid_tenant_key(client: httpx.AsyncClient, completion: Body) -> None:
    missing = await client.post("/v1/chat/completions", json=completion())
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == 'Bearer realm="model-router"'
    assert missing.json()["error"]["type"] == "authentication_error"

    wrong = await client.post(
        "/v1/chat/completions",
        json=completion(),
        headers={"Authorization": "Bearer nope-nope-nope-nope"},
    )
    assert wrong.status_code == 401
    assert 'error="invalid_token"' in wrong.headers["www-authenticate"]


@pytest.mark.parametrize(
    ("payload", "message_part"),
    [
        ({"stream": True, "messages": [{"role": "user", "content": "x"}]}, "Streaming"),
        ({"model": "m"}, "messages"),
        ({"messages": []}, "non-empty"),
        ({"messages": [{"content": "no role"}]}, "role"),
        ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "max_tokens"),
        ({"messages": [{"role": "user", "content": "x"}], "max_tokens": "10"}, "max_tokens"),
        ({"messages": [{"role": "user", "content": "x"}], "max_tokens": True}, "max_tokens"),
        ({"prompt": 5}, "prompt"),
        ([1, 2], "JSON object"),
        ({"model": "bad model", "messages": [{"role": "user", "content": "x"}]}, "model"),
        ({"model": "m" * 257, "messages": [{"role": "user", "content": "x"}]}, "model"),
        ({"model": "m\n", "messages": [{"role": "user", "content": "x"}]}, "model"),
        ({"messages": [{"role": "user", "content": "x"}] * 4097}, "at most 4096"),
        ({"messages": [{"role": "user", "content": [{"type": "text"}] * 4097}]}, "at most 4096"),
    ],
)
async def test_request_validation(
    client: httpx.AsyncClient, tenant_a: dict[str, str], payload: Any, message_part: str
) -> None:
    response = await client.post("/v1/chat/completions", json=payload, headers=tenant_a)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request"
    assert message_part in error["message"]
    assert error["request_id"] == response.headers["x-request-id"]


async def test_malformed_and_oversized_bodies(
    app_for: Any, make_settings: Settings, tenant_a: dict[str, str]
) -> None:
    async with app_for(make_settings(max_body_bytes=256)) as client:
        bad = await client.post(
            "/v1/chat/completions",
            content=b"{",
            headers={**tenant_a, "Content-Type": "application/json"},
        )
        big = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x" * 1000}]},
            headers=tenant_a,
        )
    assert bad.status_code == 400
    assert big.status_code == 413
    assert big.json()["error"]["type"] == "request_too_large"


async def test_deeply_nested_json_is_invalid_not_an_internal_error(
    client: httpx.AsyncClient, tenant_a: dict[str, str]
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        content=b"[" * 100_000,
        headers={**tenant_a, "Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"


async def test_success_passes_the_provider_body_and_sets_headers(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    response = await client.post(
        "/v1/chat/completions", json=completion(), headers={**tenant_a, "X-Request-Id": "trace-1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"].startswith("Hello from primary")
    assert response.headers["x-model-router-provider"] == "primary"
    assert response.headers["x-request-id"] == "trace-1"
    assert response.headers["x-ratelimit-limit"] == "50000"
    assert int(response.headers["x-ratelimit-remaining"]) == 50_000 - body["usage"]["total_tokens"]
    assert response.headers["x-ratelimit-window-seconds"] == "60"


async def test_failover_is_visible_in_the_provider_header(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json=completion(primary={"behavior": "rate_limited"}),
        headers=tenant_a,
    )
    assert response.status_code == 200
    assert response.headers["x-model-router-provider"] == "secondary"


async def test_rate_limit_exceeded_response(
    client: httpx.AsyncClient, tenant_b: dict[str, str], completion: Body
) -> None:
    # Tenant B has a 2000 token window; every call reserves ~960 tokens (prompt estimate plus
    # max_tokens) and is settled to ~950 reported tokens.
    body = completion(max_tokens=950, completion_tokens=940)
    first = await client.post("/v1/chat/completions", json=body, headers=tenant_b)
    second = await client.post("/v1/chat/completions", json=body, headers=tenant_b)
    assert first.status_code == 200
    assert second.status_code == 200
    remaining = int(second.headers["x-ratelimit-remaining"])
    assert 0 <= remaining < 200

    third = await client.post("/v1/chat/completions", json=body, headers=tenant_b)
    assert third.status_code == 429
    error = third.json()["error"]
    assert error["type"] == "rate_limit_exceeded"
    retry_after = int(third.headers["retry-after"])
    assert retry_after >= 1
    assert error["retry_after_seconds"] == retry_after
    assert third.headers["x-ratelimit-limit"] == str(TENANT_B_LIMIT)
    assert third.headers["x-ratelimit-remaining"] == str(remaining)


async def test_request_that_can_never_fit_the_window(
    client: httpx.AsyncClient, tenant_b: dict[str, str], completion: Body
) -> None:
    response = await client.post(
        "/v1/chat/completions", json=completion(max_tokens=5000), headers=tenant_b
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "request_too_large"
    assert "retry-after" not in response.headers


async def test_all_providers_down_is_a_standardised_503(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    response = await client.post(
        "/v1/chat/completions", json=completion(behavior="error"), headers=tenant_a
    )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    error = response.json()["error"]
    assert set(error) == {"type", "message", "request_id"}
    assert error["type"] == "upstream_unavailable"
    assert not any(marker in response.text for marker in INTERNAL_MARKERS)


async def test_provider_client_error_is_relayed_sanitised(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json=completion(primary={"behavior": "bad_request"}),
        headers=tenant_a,
    )
    assert response.status_code == 400
    assert response.json()["error"] == {
        "type": "upstream_rejected",
        "message": "max_tokens is too large for this model",
        "request_id": response.headers["x-request-id"],
    }


async def test_provider_credential_failure_is_the_gateways_problem(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json=completion(primary={"behavior": "unauthorized"}),
        headers=tenant_a,
    )
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_unavailable"
    assert "Incorrect API key" not in response.text


async def test_unexpected_exceptions_never_leak(
    client: httpx.AsyncClient,
    tenant_a: dict[str, str],
    completion: Body,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def explode(self: ModelRouter, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(ModelRouter, "complete", explode)
    response = await client.post("/v1/chat/completions", json=completion(), headers=tenant_a)
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "internal_error"
    assert "secret internal detail" not in response.text


async def test_usage_and_health_endpoints(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body, db_path: Path
) -> None:
    await client.post(
        "/v1/chat/completions", json=completion(completion_tokens=100), headers=tenant_a
    )
    usage = await client.get("/v1/usage", headers=tenant_a)
    assert usage.status_code == 200
    body = usage.json()
    assert body["tenant_id"] == "tenant-a"
    assert body["used"] > 100
    assert body["remaining"] == 50_000 - body["used"]
    assert (await client.get("/v1/usage")).status_code == 401

    health = await client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}  # liveness discloses nothing about the deployment
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ok", "ledger": "ok"}
    assert str(db_path) not in ready.text  # no file system paths for anonymous callers
    assert db_path.exists()
    assert db_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Ledger failures
# ---------------------------------------------------------------------------


async def test_ledger_outage_fails_closed_before_any_provider_call(
    client: httpx.AsyncClient,
    tenant_a: dict[str, str],
    completion: Body,
    provider_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def locked(self: TokenLedgerStore, *args: Any, **kwargs: Any) -> Any:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(TokenLedgerStore, "try_reserve", locked)
    response = await client.post("/v1/chat/completions", json=completion(), headers=tenant_a)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    error = response.json()["error"]
    assert error["type"] == "service_unavailable"
    assert error["request_id"] == response.headers["x-request-id"]
    assert "locked" not in response.text
    assert provider_state("primary").requests == 0

    usage = await client.get("/v1/usage", headers=tenant_a)
    assert usage.status_code == 200  # reads still work; only admission is broken here


async def test_settlement_failure_keeps_the_served_response_and_the_estimate(
    client: httpx.AsyncClient,
    tenant_a: dict[str, str],
    completion: Body,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(self: TokenLedgerStore, *args: Any, **kwargs: Any) -> Any:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(TokenLedgerStore, "settle", broken)
    response = await client.post(
        "/v1/chat/completions",
        json=completion(max_tokens=400, completion_tokens=10),
        headers=tenant_a,
    )
    assert response.status_code == 200
    assert response.headers["x-model-router-provider"] == "primary"
    usage = await client.get("/v1/usage", headers=tenant_a)
    # Settlement to the real (small) usage failed, so the conservative estimate stands.
    assert usage.json()["used"] >= 400


def test_main_exits_cleanly_when_the_database_cannot_be_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied")
    env = {
        "MODEL_ROUTER_PRIMARY_URL": "http://127.0.0.1:1",
        "MODEL_ROUTER_SECONDARY_URL": "http://127.0.0.1:2",
        "MODEL_ROUTER_TENANTS_JSON": json.dumps({"tenants": {"k" * 40: {"tenant_id": "t"}}}),
        "MODEL_ROUTER_DATABASE_PATH": str(blocker / "ledger.sqlite3"),
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "model_router.__main__.uvicorn.run",
        lambda *a, **k: pytest.fail("the server must not start without a ledger"),
    )
    assert main([]) == 2


# ---------------------------------------------------------------------------
# Round two: request ids, provider oddities, validation, health, routing
# ---------------------------------------------------------------------------


async def test_reused_request_ids_do_not_collide_in_the_ledger(
    client: httpx.AsyncClient, tenant_a: dict[str, str], tenant_b: dict[str, str], completion: Body
) -> None:
    same = {"X-Request-Id": "shared-id"}
    first = await client.post(
        "/v1/chat/completions", json=completion(), headers={**tenant_a, **same}
    )
    second = await client.post(
        "/v1/chat/completions", json=completion(), headers={**tenant_a, **same}
    )
    other_tenant = await client.post(
        "/v1/chat/completions", json=completion(), headers={**tenant_b, **same}
    )
    assert [r.status_code for r in (first, second, other_tenant)] == [200, 200, 200]
    assert first.headers["x-request-id"] == "shared-id"


@pytest.mark.parametrize("behavior", ["request_timeout", "bad_encoding", "nan"])
async def test_odd_primary_answers_fail_over_instead_of_erroring(
    client: httpx.AsyncClient,
    tenant_a: dict[str, str],
    completion: Body,
    provider_state: Any,
    behavior: str,
) -> None:
    response = await client.post(
        "/v1/chat/completions", json=completion(primary={"behavior": behavior}), headers=tenant_a
    )
    assert response.status_code == 200
    assert response.headers["x-model-router-provider"] == "secondary"
    assert provider_state("primary").requests == 1


async def test_text_json_content_type_is_accepted(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    response = await client.post(
        "/v1/chat/completions", json=completion(primary={"behavior": "text_json"}), headers=tenant_a
    )
    assert response.status_code == 200
    assert response.headers["x-model-router-provider"] == "primary"


async def test_leaky_provider_messages_are_replaced_by_a_generic_one(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json=completion(primary={"behavior": "leaky_bad_request"}),
        headers=tenant_a,
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error == {
        "type": "upstream_rejected",
        "message": "The model provider rejected the request",
        "request_id": response.headers["x-request-id"],
    }
    for marker in ("10.0.0.5", "/srv/", "sk-live"):
        assert marker not in response.text


@pytest.mark.parametrize(
    ("payload", "message_part"),
    [
        ({"messages": [{"role": "user", "content": "x"}], "stream": 1}, "boolean"),
        ({"messages": [{"role": "user", "content": "x"}], "stream": "true"}, "boolean"),
        ({"messages": [{"role": "user", "content": "x"}], "n": 0}, "'n'"),
        ({"messages": [{"role": "user", "content": {"text": "x"}}]}, "content"),
        ({"messages": [{"role": "user", "content": ["x"]}]}, "content part"),
    ],
)
async def test_round_two_request_validation(
    client: httpx.AsyncClient, tenant_a: dict[str, str], payload: Any, message_part: str
) -> None:
    response = await client.post("/v1/chat/completions", json=payload, headers=tenant_a)
    assert response.status_code == 400
    assert message_part in response.json()["error"]["message"]


async def test_odd_content_parts_do_not_crash_the_estimator(
    client: httpx.AsyncClient, tenant_a: dict[str, str]
) -> None:
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": None}, {"type": "image"}]}
        ],
        "max_tokens": 10,
    }
    response = await client.post("/v1/chat/completions", json=payload, headers=tenant_a)
    assert response.status_code == 200


async def test_n_multiplies_the_reserved_completion_budget(
    client: httpx.AsyncClient, tenant_b: dict[str, str], completion: Body
) -> None:
    # tenant-b has a 2000 token window: 3 x 900 completions can never fit.
    response = await client.post(
        "/v1/chat/completions",
        json={**completion(max_tokens=900), "n": 3},
        headers=tenant_b,
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "request_too_large"


async def test_health_reports_a_broken_ledger(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: TokenLedgerStore) -> None:
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(TokenLedgerStore, "probe", broken)
    ready = await client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["status"] == "degraded"
    assert ready.json()["ledger"] == "unavailable"
    assert (await client.get("/healthz")).status_code == 200  # the process itself is alive


async def test_routing_errors_use_the_standard_error_shape(
    client: httpx.AsyncClient, tenant_a: dict[str, str]
) -> None:
    missing = await client.get("/nope", headers=tenant_a)
    assert missing.status_code == 404
    assert missing.json()["error"]["type"] == "not_found"
    wrong = await client.put("/v1/chat/completions", headers=tenant_a)
    assert wrong.status_code == 405
    assert wrong.json()["error"]["type"] == "method_not_allowed"
    assert "POST" in wrong.headers["allow"]
    trailing = await client.post("/v1/chat/completions/", json={}, headers=tenant_a)
    assert trailing.status_code == 404


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_read_only_database_is_rejected_at_startup(tmp_path: Path) -> None:
    import os
    import stat

    path = tmp_path / "ledger.sqlite3"
    TokenLedgerStore(path).close()
    for name in (path, tmp_path):
        os.chmod(name, stat.S_IRUSR | stat.S_IXUSR if name == tmp_path else stat.S_IRUSR)
    try:
        if os.access(path, os.W_OK):
            pytest.skip("running with privileges that ignore file modes")
        with pytest.raises(sqlite3.OperationalError):
            TokenLedgerStore(path)
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# Fine-tuned and self-hosted deployments: dialects, fallback control, capacity,
# per-tenant models, token accounting without usage
# ---------------------------------------------------------------------------

ANTHROPIC_SECONDARY = ProviderSettings(
    "secondary", "http://secondary", api_key="sk-ant", dialect="anthropic", timeout_ms=300
)


async def test_anthropic_secondary_serves_in_the_chat_completions_shape(
    app_for: Any,
    make_settings: Settings,
    tenant_a: dict[str, str],
    completion: Body,
    provider_state: Any,
) -> None:
    async with app_for(make_settings(secondary=ANTHROPIC_SECONDARY)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={**completion(primary={"behavior": "rate_limited"}), "stop": "END"},
            headers=tenant_a,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"].startswith("Hello from secondary")
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0
    assert response.headers["x-model-router-provider"] == "secondary"
    assert response.headers["x-model-router-fallback"] == "true"
    secondary = provider_state("secondary")
    assert secondary.last_path == "/v1/messages"
    assert secondary.last_headers["x-api-key"] == "sk-ant"
    assert secondary.last_headers["anthropic-version"]
    assert secondary.last_payload["max_tokens"] == 50
    assert secondary.last_payload["stop_sequences"] == ["END"]
    assert secondary.last_payload["messages"][0]["role"] == "user"


async def test_anthropic_tool_calls_round_trip(
    app_for: Any, make_settings: Settings, tenant_a: dict[str, str], completion: Body
) -> None:
    payload = {
        **completion(primary={"behavior": "error"}),
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {}}}],
    }
    controls = {"X-Mock-Controls": json.dumps({"secondary": {"behavior": "tool_use"}})}
    async with app_for(make_settings(secondary=ANTHROPIC_SECONDARY)) as client:
        response = await client.post(
            "/v1/chat/completions", json=payload, headers={**tenant_a, **controls}
        )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


async def test_anthropic_overloaded_primary_fails_over(
    app_for: Any, make_settings: Settings, tenant_a: dict[str, str], completion: Body
) -> None:
    primary = ProviderSettings("primary", "http://primary", dialect="anthropic", timeout_ms=300)
    controls = {"X-Mock-Controls": json.dumps({"primary": {"behavior": "overloaded"}})}
    async with app_for(make_settings(primary=primary)) as client:
        response = await client.post(
            "/v1/chat/completions", json=completion(), headers={**tenant_a, **controls}
        )
    assert response.status_code == 200
    assert response.headers["x-model-router-provider"] == "secondary"


async def test_shapes_no_provider_can_express_are_a_400(
    app_for: Any, make_settings: Settings, tenant_a: dict[str, str], completion: Body
) -> None:
    primary = ProviderSettings("primary", "http://primary", dialect="anthropic", timeout_ms=300)
    async with app_for(make_settings(primary=primary, secondary=ANTHROPIC_SECONDARY)) as client:
        response = await client.post(
            "/v1/chat/completions", json={**completion(), "n": 2}, headers=tenant_a
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "unsupported_by_provider"
    assert "n > 1" in error["message"]

    # A compatible primary that fails, then a secondary that cannot express the shape:
    # the request failed for provider reasons, so it is a 503, not the caller's fault.
    async with app_for(make_settings(secondary=ANTHROPIC_SECONDARY)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={**completion(primary={"behavior": "rate_limited"}), "n": 2},
            headers=tenant_a,
        )
    assert response.status_code == 503


async def test_fallback_can_be_disabled_per_request_or_by_default(
    app_for: Any,
    make_settings: Settings,
    tenant_a: dict[str, str],
    completion: Body,
    provider_state: Any,
) -> None:
    async with app_for(make_settings()) as client:
        strict = await client.post(
            "/v1/chat/completions",
            json=completion(primary={"behavior": "rate_limited"}),
            headers={**tenant_a, "X-Model-Router-Allow-Fallback": "false"},
        )
        bad_header = await client.post(
            "/v1/chat/completions",
            json=completion(),
            headers={**tenant_a, "X-Model-Router-Allow-Fallback": "maybe"},
        )
    assert strict.status_code == 503
    assert "fallback was disabled" in strict.json()["error"]["message"]
    assert provider_state("secondary").requests == 0
    assert bad_header.status_code == 400

    async with app_for(make_settings(allow_fallback_default=False)) as client:
        default_strict = await client.post(
            "/v1/chat/completions",
            json=completion(primary={"behavior": "rate_limited"}),
            headers=tenant_a,
        )
        opted_in = await client.post(
            "/v1/chat/completions",
            json=completion(primary={"behavior": "rate_limited"}),
            headers={**tenant_a, "X-Model-Router-Allow-Fallback": "yes"},
        )
    assert default_strict.status_code == 503
    assert opted_in.status_code == 200
    assert opted_in.headers["x-model-router-fallback"] == "true"


async def test_saturated_primary_fails_over_without_queueing(
    app_for: Any,
    make_settings: Settings,
    tenant_a: dict[str, str],
    completion: Body,
    provider_state: Any,
) -> None:
    primary = ProviderSettings(
        "primary", "http://primary", model="primary-model", timeout_ms=2000, max_concurrency=1
    )
    async with app_for(make_settings(primary=primary)) as client:
        first = asyncio.ensure_future(
            client.post(
                "/v1/chat/completions",
                json=completion(primary={"behavior": "slow", "delay_ms": 400}),
                headers=tenant_a,
            )
        )
        await asyncio.sleep(0.05)
        started = time.perf_counter()
        second = await client.post("/v1/chat/completions", json=completion(), headers=tenant_a)
        elapsed = time.perf_counter() - started
        first_response = await first
    assert second.status_code == 200
    assert second.headers["x-model-router-provider"] == "secondary"
    assert elapsed < 0.3  # it did not wait for the primary's slot
    assert first_response.headers["x-model-router-provider"] == "primary"
    assert provider_state("primary").requests == 1


async def test_everything_saturated_is_a_503_with_a_short_retry(
    app_for: Any, make_settings: Settings, tenant_a: dict[str, str], completion: Body
) -> None:
    primary = ProviderSettings("primary", "http://primary", timeout_ms=2000, max_concurrency=1)
    secondary = ProviderSettings(
        "secondary", "http://secondary", timeout_ms=2000, max_concurrency=1
    )
    async with app_for(make_settings(primary=primary, secondary=secondary)) as client:
        slow = completion(behavior="slow", delay_ms=400)
        holders = [
            asyncio.ensure_future(client.post("/v1/chat/completions", json=slow, headers=tenant_a))
            for _ in range(2)
        ]
        await asyncio.sleep(0.05)
        third = await client.post("/v1/chat/completions", json=completion(), headers=tenant_a)
        await asyncio.gather(*holders)
    assert third.status_code == 503
    assert third.headers["retry-after"] == "1"
    assert "capacity" in third.json()["error"]["message"]


async def test_retry_after_hints_put_the_provider_on_cooldown(
    client: httpx.AsyncClient,
    tenant_a: dict[str, str],
    completion: Body,
    provider_state: Any,
    clock: Any,
) -> None:
    # The mock answers 429 with Retry-After: 1, so the primary is paused for a second.
    first = await client.post(
        "/v1/chat/completions",
        json=completion(primary={"behavior": "rate_limited"}),
        headers=tenant_a,
    )
    second = await client.post("/v1/chat/completions", json=completion(), headers=tenant_a)
    assert first.headers["x-model-router-provider"] == "secondary"
    assert second.headers["x-model-router-provider"] == "secondary"
    assert provider_state("primary").requests == 1  # not contacted during the cooldown
    clock.advance(1.5)
    third = await client.post("/v1/chat/completions", json=completion(), headers=tenant_a)
    assert third.headers["x-model-router-provider"] == "primary"


async def test_tenant_model_pins_and_allow_lists(
    app_for: Any, make_settings: Settings, completion: Body, provider_state: Any
) -> None:
    tenants = {
        TENANT_A_KEY: TenantSettings(
            "tenant-a", model="tenant-a-ft", allowed_models=("tenant-a-ft", "general")
        ),
        TENANT_B_KEY: TenantSettings("tenant-b", allowed_models=("general",)),
    }
    headers_a = {"Authorization": f"Bearer {TENANT_A_KEY}"}
    headers_b = {"Authorization": f"Bearer {TENANT_B_KEY}"}
    async with app_for(make_settings(tenants=tenants)) as client:
        pinned = await client.post(
            "/v1/chat/completions", json={**completion(), "model": "general"}, headers=headers_a
        )
        pinned_model_seen = provider_state("primary").last_payload["model"]
        denied = await client.post(
            "/v1/chat/completions", json={**completion(), "model": "other"}, headers=headers_a
        )
        provider_pin = await client.post(
            "/v1/chat/completions", json={**completion(), "model": "general"}, headers=headers_b
        )
        usage = await client.get("/v1/usage", headers=headers_a)
    assert pinned.status_code == 200
    assert pinned.headers["x-model-router-model"] == "tenant-a-ft"  # tenant pin beats provider pin
    assert pinned_model_seen == "tenant-a-ft"
    assert denied.status_code == 400
    assert denied.json()["error"]["type"] == "model_not_allowed"
    assert provider_pin.status_code == 200
    assert provider_pin.headers["x-model-router-model"] == "primary-model"
    assert usage.json()["model"] == "tenant-a-ft"
    assert usage.json()["allowed_models"] == ["tenant-a-ft", "general"]


async def test_missing_usage_is_settled_from_the_response_not_the_budget(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json=completion(max_tokens=800, primary={"behavior": "no_usage"}),
        headers=tenant_a,
    )
    assert response.status_code == 200
    assert response.headers["x-model-router-tokens-estimated"] == "true"
    usage = await client.get("/v1/usage", headers=tenant_a)
    assert 0 < usage.json()["used"] < 200  # the short answer, not the 800 token budget


async def test_calibration_is_learned_from_reported_usage_and_exposed(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    # The mock reports prompt tokens as len(json(messages)) / 4, which exceeds the
    # heuristic's estimate; after a few requests the primary's factor has moved.
    long_prompt = completion()
    long_prompt["messages"] = [{"role": "user", "content": "word " * 200}]
    for _ in range(3):
        assert (
            await client.post("/v1/chat/completions", json=long_prompt, headers=tenant_a)
        ).status_code == 200
    usage = await client.get("/v1/usage", headers=tenant_a)
    factors = usage.json()["calibration"]
    assert set(factors) == {"primary", "secondary"}
    assert factors["primary"] != 1.0
    assert factors["secondary"] == 1.0


def test_unknown_token_estimator_fails_at_startup(
    make_settings: Settings, store: TokenLedgerStore, http_client: httpx.AsyncClient
) -> None:
    from model_router.app import build_limiter, build_router
    from model_router.config import ConfigurationError

    settings = make_settings(token_estimator="tiktoken:")
    with pytest.raises(ConfigurationError, match="TOKEN_ESTIMATOR"):
        build_router(settings, build_limiter(settings, store), http_client)


async def test_allow_list_without_a_pin_requires_an_explicit_model(
    app_for: Any, make_settings: Settings, completion: Body
) -> None:
    tenants = {TENANT_B_KEY: TenantSettings("tenant-b", allowed_models=("general",))}
    headers = {"Authorization": f"Bearer {TENANT_B_KEY}"}
    async with app_for(make_settings(tenants=tenants)) as client:
        body = completion()
        del body["model"]
        anonymous = await client.post("/v1/chat/completions", json=body, headers=headers)
        named = await client.post(
            "/v1/chat/completions", json={**body, "model": "general"}, headers=headers
        )
    assert anonymous.status_code == 400
    assert anonymous.json()["error"]["type"] == "model_required"
    assert named.status_code == 200


async def test_admission_refusals_do_not_grow_the_request_log(
    client: httpx.AsyncClient, tenant_b: dict[str, str], completion: Body, store: TokenLedgerStore
) -> None:
    served = await client.post("/v1/chat/completions", json=completion(), headers=tenant_b)
    assert served.status_code == 200
    for _ in range(5):
        refused = await client.post(
            "/v1/chat/completions", json=completion(max_tokens=1960), headers=tenant_b
        )
        assert refused.status_code == 429
    too_large = await client.post(
        "/v1/chat/completions", json=completion(max_tokens=5000), headers=tenant_b
    )
    assert too_large.status_code == 400
    assert store.request_log_count("tenant-b") == 1  # only the request a provider served


async def test_model_names_in_headers_are_bounded(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    model = "org/" + "m" * 230 + ":tag@v1.2+beta"
    response = await client.post(
        "/v1/chat/completions", json={**completion(), "model": model}, headers=tenant_a
    )
    assert response.status_code == 200
    assert response.headers["x-model-router-model"] == "primary-model"  # the provider pin wins


async def test_request_ids_with_trailing_newlines_are_replaced(
    client: httpx.AsyncClient, tenant_a: dict[str, str], completion: Body
) -> None:
    from model_router.app import resolve_request_id

    assert resolve_request_id("abc\n") != "abc\n"
    assert resolve_request_id("abc") == "abc"
