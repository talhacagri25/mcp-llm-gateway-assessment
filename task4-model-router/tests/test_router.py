"""Failover mechanics, timeout races and token accounting of ``ModelRouter``."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from model_router.app import build_limiter, build_router
from model_router.config import ProviderSettings, RouterSettings
from model_router.providers import ProviderClient, ProviderResponse
from model_router.rate_limiter import RateLimitExceededError
from model_router.router import (
    AllProvidersFailedError,
    ModelRouter,
    UpstreamRejectedError,
    sanitized_provider_message,
)
from model_router.storage import TokenLedgerStore

Settings = Callable[..., RouterSettings]
Body = Callable[..., dict[str, Any]]


@pytest.fixture
def make_router(
    make_settings: Settings, store: TokenLedgerStore, http_client: httpx.AsyncClient, clock: Any
) -> Callable[..., tuple[ModelRouter, Any]]:
    def factory(**overrides: Any) -> tuple[ModelRouter, Any]:
        settings = make_settings(**overrides)
        limiter = build_limiter(settings, store, clock=clock)
        return build_router(settings, limiter, http_client, clock=clock), limiter

    return factory


async def test_primary_success_is_settled_to_reported_usage(
    make_router: Any, completion: Body, provider_state: Any
) -> None:
    router, limiter = make_router()
    payload = completion(max_tokens=500, completion_tokens=40)
    result = await router.complete("tenant-a", payload, "req-1")
    assert result.provider == "primary"
    assert [a.outcome for a in result.attempts] == ["success"]
    assert result.tokens_reserved == router.estimate_reservation("tenant-a", payload) >= 500
    assert result.tokens_charged == result.body["usage"]["total_tokens"]
    assert (await limiter.snapshot("tenant-a")).used == result.tokens_charged
    assert provider_state("secondary").requests == 0


@pytest.mark.parametrize(
    ("behavior", "expected_outcome"),
    [
        ("rate_limited", "rate_limited"),
        ("error", "server_error"),
        ("garbage", "unparseable"),
        ("slow", "timeout"),
    ],
)
async def test_failover_to_secondary(
    make_router: Any, completion: Body, provider_state: Any, behavior: str, expected_outcome: str
) -> None:
    router, _ = make_router()
    payload = completion(primary={"behavior": behavior, "delay_ms": 2000})
    started = time.perf_counter()
    result = await router.complete("tenant-a", payload, "req-1")
    elapsed = time.perf_counter() - started
    assert result.provider == "secondary"
    assert [a.outcome for a in result.attempts] == [expected_outcome, "success"]
    assert provider_state("primary").requests == 1
    assert provider_state("secondary").requests == 1
    if behavior == "slow":
        # The 2 s provider sleep is abandoned at the 300 ms deadline.
        assert 0.25 < elapsed < 1.0


async def test_timeout_cancels_the_slow_attempt(make_router: Any, completion: Body) -> None:
    router, _ = make_router()
    payload = completion(primary={"behavior": "slow", "delay_ms": 5000})
    started = time.perf_counter()
    result = await router.complete("tenant-a", payload, "req-1")
    assert result.provider == "secondary"
    assert time.perf_counter() - started < 1.0
    assert result.attempts[0].latency_ms is not None
    assert result.attempts[0].latency_ms < 1000


async def test_client_error_is_not_failed_over_and_releases_budget(
    make_router: Any, completion: Body, provider_state: Any
) -> None:
    router, limiter = make_router()
    with pytest.raises(UpstreamRejectedError) as exc_info:
        await router.complete("tenant-a", completion(primary={"behavior": "bad_request"}), "req-1")
    assert exc_info.value.status == 400
    assert exc_info.value.message == "max_tokens is too large for this model"
    assert provider_state("secondary").requests == 0
    assert (await limiter.snapshot("tenant-a")).used == 0


async def test_all_providers_failing_releases_the_reservation(
    make_router: Any, completion: Body, store: TokenLedgerStore
) -> None:
    router, limiter = make_router()
    with pytest.raises(AllProvidersFailedError) as exc_info:
        await router.complete("tenant-a", completion(behavior="error"), "req-1")
    assert [a.outcome for a in exc_info.value.attempts] == ["server_error", "server_error"]
    assert (await limiter.snapshot("tenant-a")).used == 0
    assert store.ledger_rows("tenant-a")[0][3] == "released"


async def test_rate_limit_is_enforced_before_calling_any_provider(
    make_router: Any, completion: Body, provider_state: Any
) -> None:
    router, _ = make_router(tokens_per_minute=100)
    with pytest.raises(RateLimitExceededError):
        await router.complete("tenant-a", completion(max_tokens=200), "req-1")
    assert provider_state("primary").requests == 0
    assert provider_state("secondary").requests == 0


async def test_circuit_breaker_skips_a_repeatedly_failing_primary(
    make_router: Any, completion: Body, provider_state: Any, clock: Any
) -> None:
    router, _ = make_router()
    for index in range(3):
        result = await router.complete(
            "tenant-a", completion(primary={"behavior": "error"}), f"r{index}"
        )
        assert result.provider == "secondary"
    assert provider_state("primary").requests == 3

    skipped = await router.complete("tenant-a", completion(), "r-skip")
    assert skipped.provider == "secondary"
    assert [a.outcome for a in skipped.attempts] == ["circuit_open", "success"]
    assert provider_state("primary").requests == 3  # not contacted while the circuit is open

    clock.advance(31)
    recovered = await router.complete("tenant-a", completion(), "r-trial")
    assert recovered.provider == "primary"
    assert provider_state("primary").requests == 4


async def test_model_rewrite_and_credentials_are_per_provider(
    make_router: Any, completion: Body, provider_state: Any
) -> None:
    router, _ = make_router()
    await router.complete("tenant-a", completion(), "req-1")
    primary = provider_state("primary")
    assert primary.last_payload["model"] == "primary-model"
    assert primary.last_headers["authorization"] == "Bearer primary-key"
    assert primary.last_headers["x-request-id"] == "req-1"

    await router.complete("tenant-a", completion(primary={"behavior": "error"}), "req-2")
    secondary = provider_state("secondary")
    assert secondary.last_payload["model"] == "client-model"
    assert "authorization" not in secondary.last_headers


async def test_cancellation_mid_flight_releases_the_reservation(
    make_router: Any, completion: Body, store: TokenLedgerStore
) -> None:
    router, limiter = make_router()
    task = asyncio.create_task(
        router.complete(
            "tenant-a", completion(primary={"behavior": "slow", "delay_ms": 5000}), "r1"
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await limiter.snapshot("tenant-a")).used == 0
    assert store.ledger_rows("tenant-a")[0][3] == "released"


def test_sanitized_provider_message() -> None:
    def response(body: Any) -> ProviderResponse:
        return ProviderResponse(status=400, body=body, raw_length=0, latency_ms=0.0)

    assert sanitized_provider_message(None) == "The model provider rejected the request"
    assert (
        sanitized_provider_message(response({"error": {"message": "too  long\nprompt"}}))
        == "too long prompt"
    )
    assert sanitized_provider_message(response({"error": "plain string"})) == "plain string"
    assert sanitized_provider_message(response({"message": "top level"})) == "top level"
    traceback = 'boom\nTraceback (most recent call last):\n  File "x.py", line 1'
    assert sanitized_provider_message(response({"error": {"message": traceback}})) == (
        "The model provider rejected the request"
    )
    assert sanitized_provider_message(response({"error": {"message": "x" * 301}})) == (
        "The model provider rejected the request"
    )
    assert sanitized_provider_message(response({"error": {"message": 5}})) == (
        "The model provider rejected the request"
    )
    # Anything that could carry internals is replaced wholesale, not trimmed.
    for leaky in (
        "see https://internal.example/docs for details",
        "handler /srv/gateway/app.py:88 failed",
        "contact ops@example.com",
        "key sk-live-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 is invalid",
        "Model `gpt-4o` does <not> exist",
    ):
        assert sanitized_provider_message(response({"error": {"message": leaky}})) == (
            "The model provider rejected the request"
        ), leaky
    assert (
        sanitized_provider_message(response({"error": {"message": "Unsupported value: 'x'."}}))
        == "Unsupported value: 'x'."
    )


async def test_abandoned_half_open_trial_does_not_lock_the_primary_out(
    make_router: Any, completion: Body, provider_state: Any, clock: Any, monkeypatch: Any
) -> None:
    router, _ = make_router()
    for index in range(3):
        await router.complete("tenant-a", completion(primary={"behavior": "error"}), f"r{index}")
    clock.advance(31)  # half open: the next call is the trial

    from model_router.providers import ProviderClient

    original = ProviderClient.complete

    async def crash_once(self: ProviderClient, *args: Any, **kwargs: Any) -> Any:
        monkeypatch.setattr(ProviderClient, "complete", original)
        raise RuntimeError("unexpected failure inside the trial")

    monkeypatch.setattr(ProviderClient, "complete", crash_once)
    with pytest.raises(RuntimeError):
        await router.complete("tenant-a", completion(), "trial")
    # Without abandon_trial the breaker would stay half open with a phantom trial
    # in flight and every later request would skip the (now healthy) primary.
    recovered = await router.complete("tenant-a", completion(), "after-trial")
    assert recovered.provider == "primary"
    assert [a.outcome for a in recovered.attempts] == ["success"]


def test_stale_half_open_trials_expire() -> None:
    from model_router.circuit_breaker import CircuitBreaker

    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=10, clock=lambda: now[0])
    breaker.record_failure()
    assert not breaker.allows_request()
    now[0] = 10
    assert breaker.allows_request()  # the trial
    assert not breaker.allows_request()  # one trial at a time
    now[0] = 19
    assert not breaker.allows_request()
    now[0] = 20  # the trial never reported back: let someone else try
    assert breaker.allows_request()
    breaker.abandon_trial()
    assert breaker.allows_request()


async def test_cancellation_during_admission_releases_the_orphan_row(
    make_router: Any, completion: Body, store: TokenLedgerStore, monkeypatch: Any
) -> None:
    import time as _time

    router, limiter = make_router()
    original = TokenLedgerStore.try_reserve

    def slow_reserve(self: TokenLedgerStore, *args: Any, **kwargs: Any) -> Any:
        _time.sleep(0.2)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TokenLedgerStore, "try_reserve", slow_reserve)
    task = asyncio.create_task(router.complete("tenant-a", completion(), "cancel-me"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(50):  # the thread finishes, then the orphan is released
        await asyncio.sleep(0.02)
        if (await limiter.snapshot("tenant-a")).used == 0 and store.ledger_rows("tenant-a"):
            break
    rows = store.ledger_rows("tenant-a")
    assert [row[3] for row in rows] == ["released"]
    assert (await limiter.snapshot("tenant-a")).used == 0


async def test_cancellation_after_the_provider_answered_keeps_the_charge(
    make_router: Any, completion: Body, store: TokenLedgerStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A served request is never released, even when the task dies during settlement."""
    router, limiter = make_router()

    async def interrupted(*args: Any, **kwargs: Any) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(limiter, "settle", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await router.complete("tenant-a", completion(max_tokens=300), "r1")
    rows = store.ledger_rows("tenant-a")
    assert [row[3] for row in rows] == ["reserved"]
    assert rows[0][2] >= 300  # the estimate stays charged; nothing was handed back


def _client_for(handler: Any, **kwargs: Any) -> ProviderClient:
    settings = ProviderSettings("p", "http://provider", timeout_ms=2000)
    transport = httpx.MockTransport(handler)
    return ProviderClient(
        settings, httpx.AsyncClient(transport=transport), forward_headers=("traceparent",), **kwargs
    )


async def test_oversized_provider_bodies_are_not_buffered(completion: Body) -> None:
    big = b'{"choices": [], "padding": "' + b"x" * 20_000 + b'"}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big, headers={"content-type": "application/json"})

    capped = _client_for(handler, max_response_bytes=4096)
    response = await capped.complete(completion(), "r", model="m", default_max_tokens=10)
    assert response.status == 200
    assert response.body is None  # treated as unparseable: the router fails over
    uncapped = _client_for(handler)
    response = await uncapped.complete(completion(), "r", model="m", default_max_tokens=10)
    assert isinstance(response.body, dict)


async def test_only_printable_ascii_header_values_are_forwarded(completion: Body) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(
            200, json={"choices": []}, headers={"content-type": "application/json"}
        )

    client = _client_for(handler)
    await client.complete(
        completion(),
        "r",
        model="m",
        default_max_tokens=10,
        client_headers={"traceparent": "00-abc-def-01", "x-other": "dropped"},
    )
    assert seen["traceparent"] == "00-abc-def-01"
    assert "x-other" not in seen
    await client.complete(
        completion(),
        "r",
        model="m",
        default_max_tokens=10,
        client_headers={"traceparent": "caf\u00e9\x7f"},
    )
    assert seen["traceparent"] == "00-abc-def-01"  # the odd value was not forwarded
