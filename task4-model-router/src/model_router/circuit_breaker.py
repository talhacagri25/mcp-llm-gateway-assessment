"""A small per-provider circuit breaker.

After ``failure_threshold`` consecutive retriable failures the circuit opens
and the provider is skipped (so callers fail over immediately instead of
waiting out a timeout on a provider known to be down). After
``recovery_seconds`` a single trial request is allowed through; success closes
the circuit, failure re-opens it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self._threshold = failure_threshold
        self._recovery = recovery_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False
        self._trial_started_at: float | None = None
        self._cooldown_until: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._clock() - self._opened_at >= self._recovery:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def cool_down(self, seconds: float) -> None:
        """Skip the provider for ``seconds`` because it asked to be left alone (``Retry-After``)."""
        if seconds <= 0:
            return
        until = self._clock() + seconds
        if self._cooldown_until is None or until > self._cooldown_until:
            self._cooldown_until = until

    @property
    def cooling_down(self) -> bool:
        return self._cooldown_until is not None and self._clock() < self._cooldown_until

    def allows_request(self) -> bool:
        if self.cooling_down:
            return False
        state = self.state
        if state is CircuitState.CLOSED:
            return True
        if state is CircuitState.HALF_OPEN:
            now = self._clock()
            # A trial that never reported back (its task was cancelled, or it
            # crashed before classification) must not lock the provider out
            # forever: after one recovery period the next caller may try again.
            stale = (
                self._trial_started_at is not None
                and now - self._trial_started_at >= self._recovery
            )
            if not self._trial_in_flight or stale:
                self._trial_in_flight = True
                self._trial_started_at = now
                return True
        return False

    def abandon_trial(self) -> None:
        """The trial ended without a verdict; let the next caller run one."""
        self._trial_in_flight = False
        self._trial_started_at = None

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._cooldown_until = None
        self.abandon_trial()

    def record_failure(self) -> None:
        self._failures += 1
        self.abandon_trial()
        if self._failures >= self._threshold:
            self._opened_at = self._clock()
