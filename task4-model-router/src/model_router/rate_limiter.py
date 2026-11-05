"""Token-aware sliding window rate limiter.

Admission is decided on an *estimated* cost (prompt estimate plus the
requested completion budget) so that a burst of large requests cannot exceed
the limit while in flight. Once the provider reports actual usage the
reservation is settled down (or up) to the real number; requests that no
provider served are released so they do not consume budget.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from model_router.storage import AdmissionResult, TokenLedgerStore

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class LedgerUnavailableError(Exception):
    """The SQLite ledger could not be read or written (locked, read-only, corrupt or full).

    Admission cannot be decided without the ledger, so callers fail closed and
    answer with a retryable 503 instead of guessing.
    """


async def _in_thread(func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except sqlite3.Error as exc:
        raise LedgerUnavailableError(f"{type(exc).__name__}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Reservation:
    tenant_id: str
    request_id: str
    tokens: int
    limit: int
    used_before: int

    @property
    def remaining_after(self) -> int:
        return max(0, self.limit - self.used_before - self.tokens)


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    tenant_id: str
    limit: int
    used: int
    window_seconds: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


class RateLimitExceededError(Exception):
    def __init__(
        self,
        tenant_id: str,
        requested: int,
        snapshot: RateLimitSnapshot,
        retry_after_ms: int | None,
    ) -> None:
        super().__init__(f"tenant {tenant_id} exceeded {snapshot.limit} tokens per window")
        self.tenant_id = tenant_id
        self.requested = requested
        self.snapshot = snapshot
        self.retry_after_ms = retry_after_ms

    @property
    def retry_after_seconds(self) -> int | None:
        if self.retry_after_ms is None:
            return None
        return max(1, math.ceil(self.retry_after_ms / 1000))

    @property
    def can_ever_succeed(self) -> bool:
        return self.retry_after_ms is not None


class SlidingWindowTokenLimiter:
    def __init__(
        self,
        store: TokenLedgerStore,
        *,
        tokens_per_window: int,
        window_seconds: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._default_limit = tokens_per_window
        self._window_ms = window_seconds * 1000
        self._window_seconds = window_seconds
        self._clock = clock
        self._overrides: dict[str, int] = {}

    def set_tenant_limit(self, tenant_id: str, tokens_per_window: int) -> None:
        self._overrides[tenant_id] = tokens_per_window

    def limit_for(self, tenant_id: str) -> int:
        return self._overrides.get(tenant_id, self._default_limit)

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    async def reserve(self, tenant_id: str, request_id: str, tokens: int) -> Reservation:
        """Admit the request or raise ``RateLimitExceededError``. Atomic across processes."""
        limit = self.limit_for(tenant_id)
        now_ms = self._now_ms()
        admission = asyncio.ensure_future(
            _in_thread(
                self._store.try_reserve,
                tenant_id,
                request_id,
                tokens,
                now_ms=now_ms,
                window_ms=self._window_ms,
                limit=limit,
            )
        )
        try:
            # The worker thread cannot be interrupted: if this task is cancelled while
            # the INSERT commits, the row would stay reserved until the window expires.
            # Shielding lets the thread finish and the callback releases the orphan.
            result = await asyncio.shield(admission)
        except asyncio.CancelledError:
            admission.add_done_callback(
                lambda done: self._release_orphan(done, tenant_id, request_id)
            )
            raise
        if result.admitted:
            return Reservation(tenant_id, request_id, tokens, limit, result.used_before)
        snapshot = RateLimitSnapshot(tenant_id, limit, result.used_before, self._window_seconds)
        raise RateLimitExceededError(tenant_id, tokens, snapshot, result.retry_after_ms)

    def _release_orphan(
        self, done: asyncio.Future[AdmissionResult], tenant_id: str, request_id: str
    ) -> None:
        if done.cancelled() or done.exception() is not None or not done.result().admitted:
            return
        logger.warning(
            "releasing a reservation whose request was cancelled during admission",
            extra={"event": "ledger.orphan_released", "tenant_id": tenant_id},
        )
        asyncio.get_running_loop().create_task(self._release_by_id(request_id))

    async def _release_by_id(self, request_id: str) -> None:
        try:
            await _in_thread(self._store.release, request_id)
        except LedgerUnavailableError:
            logger.exception(
                "orphan release failed", extra={"event": "ledger.orphan_release_failed"}
            )

    async def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        await _in_thread(self._store.settle, reservation.request_id, actual_tokens)

    async def release(self, reservation: Reservation) -> None:
        await _in_thread(self._store.release, reservation.request_id)

    async def snapshot(self, tenant_id: str) -> RateLimitSnapshot:
        used = await _in_thread(
            self._store.window_usage, tenant_id, now_ms=self._now_ms(), window_ms=self._window_ms
        )
        return RateLimitSnapshot(tenant_id, self.limit_for(tenant_id), used, self._window_seconds)

    async def evict_expired(self) -> int:
        return await _in_thread(
            self._store.evict_expired,
            now_ms=self._now_ms(),
            window_ms=self._window_ms,
            keep_log_ms=24 * 60 * 60 * 1000,
        )
