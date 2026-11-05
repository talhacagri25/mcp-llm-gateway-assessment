"""Sliding window semantics with a controllable clock."""

from __future__ import annotations

import asyncio

import pytest

from model_router.rate_limiter import RateLimitExceededError, SlidingWindowTokenLimiter
from model_router.storage import TokenLedgerStore

from .conftest import FakeClock


def make_limiter(
    store: TokenLedgerStore, clock: FakeClock, limit: int = 50_000
) -> SlidingWindowTokenLimiter:
    return SlidingWindowTokenLimiter(store, tokens_per_window=limit, window_seconds=60, clock=clock)


async def test_window_slides_with_time(store: TokenLedgerStore, clock: FakeClock) -> None:
    limiter = make_limiter(store, clock)
    first = await limiter.reserve("t", "r1", 30_000)
    assert first.remaining_after == 20_000
    clock.advance(10)
    await limiter.reserve("t", "r2", 20_000)

    with pytest.raises(RateLimitExceededError) as exc_info:
        await limiter.reserve("t", "r3", 1)
    exc = exc_info.value
    assert exc.snapshot.used == 50_000
    assert exc.snapshot.remaining == 0
    assert exc.retry_after_seconds == 50  # r1 (30k) leaves the window 60 s after it was admitted
    assert exc.can_ever_succeed

    clock.advance(49.5)
    with pytest.raises(RateLimitExceededError):
        await limiter.reserve("t", "r4", 1)
    clock.advance(0.6)
    reservation = await limiter.reserve("t", "r5", 30_000)
    assert reservation.used_before == 20_000


async def test_settlement_and_release_return_budget(
    store: TokenLedgerStore, clock: FakeClock
) -> None:
    limiter = make_limiter(store, clock)
    reservation = await limiter.reserve("t", "r1", 40_000)
    await limiter.settle(reservation, 1_000)
    assert (await limiter.snapshot("t")).used == 1_000
    second = await limiter.reserve("t", "r2", 45_000)
    await limiter.release(second)
    assert (await limiter.snapshot("t")).used == 1_000


async def test_per_tenant_override(store: TokenLedgerStore, clock: FakeClock) -> None:
    limiter = make_limiter(store, clock)
    limiter.set_tenant_limit("small", 100)
    assert limiter.limit_for("small") == 100
    assert limiter.limit_for("other") == 50_000
    await limiter.reserve("small", "r1", 100)
    with pytest.raises(RateLimitExceededError):
        await limiter.reserve("small", "r2", 1)
    await limiter.reserve("other", "r3", 49_000)


async def test_request_bigger_than_the_limit_can_never_succeed(
    store: TokenLedgerStore, clock: FakeClock
) -> None:
    limiter = make_limiter(store, clock, limit=1_000)
    with pytest.raises(RateLimitExceededError) as exc_info:
        await limiter.reserve("t", "r1", 1_001)
    assert exc_info.value.can_ever_succeed is False
    assert exc_info.value.retry_after_seconds is None


async def test_concurrent_coroutines_cannot_overshoot(
    store: TokenLedgerStore, clock: FakeClock
) -> None:
    limiter = make_limiter(store, clock)
    results = await asyncio.gather(
        *(limiter.reserve("t", f"r{i}", 2_000) for i in range(40)), return_exceptions=True
    )
    admitted = [r for r in results if not isinstance(r, BaseException)]
    rejected = [r for r in results if isinstance(r, RateLimitExceededError)]
    assert len(admitted) == 25
    assert len(rejected) == 15
    assert (await limiter.snapshot("t")).used == 50_000


async def test_eviction_removes_expired_rows(store: TokenLedgerStore, clock: FakeClock) -> None:
    limiter = make_limiter(store, clock)
    await limiter.reserve("t", "r1", 10)
    clock.advance(61)
    assert await limiter.evict_expired() == 1
    assert store.ledger_rows("t") == []
