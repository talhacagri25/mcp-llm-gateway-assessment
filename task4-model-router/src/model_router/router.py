"""Failover routing with token accounting.

``ModelRouter.complete`` reserves the estimated token cost of a request in
the sliding window, tries the primary provider under its deadline, falls over
to the secondary on 429, timeout, transport failure, server error, saturation
or an unsupported request shape, settles the reservation with the provider's
reported usage (or an estimate when none is reported), and releases it when
no provider served the request. Every exceptional outcome is a typed
exception carrying only what the HTTP layer needs to build a sanitised error
payload.
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from model_router.circuit_breaker import CircuitBreaker
from model_router.dialects import UnsupportedRequestError
from model_router.providers import (
    ProviderClient,
    ProviderResponse,
    ProviderSaturatedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    parse_retry_after,
)
from model_router.rate_limiter import Reservation, SlidingWindowTokenLimiter
from model_router.token_estimator import (
    Calibration,
    HeuristicTokenEstimator,
    TokenEstimator,
    completion_tokens_from_body,
    prompt_tokens,
    reported_usage,
    requested_choices,
    requested_completion_tokens,
)

logger = logging.getLogger(__name__)

_SAFE_MESSAGE_RE: Final = re.compile(r"^[A-Za-z0-9 `'\".,:;!?()\[\]_+=%-]{1,200}$")
"""Letters, digits, spaces and plain punctuation only: no slashes (paths, URLs), no ``@``."""
_MAX_MESSAGE_TOKEN_LENGTH: Final = 32
"""A word longer than this is an identifier, hash or key, never something a caller needs."""
LOCAL_OUTCOMES: Final = frozenset({"circuit_open", "saturated", "unsupported", "fallback_disabled"})
"""Outcomes decided without contacting the provider: they never count as provider failures,
and a request that only produced these cost nothing worth persisting."""
_LOCAL_OUTCOMES = LOCAL_OUTCOMES


@dataclass(frozen=True, slots=True)
class Attempt:
    provider: str
    outcome: str
    status: int | None = None
    latency_ms: float | None = None
    model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "outcome": self.outcome,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class RoutingResult:
    body: dict[str, Any]
    provider: str
    model: str | None
    fallback: bool
    """True when a provider other than the primary served the request."""
    attempts: tuple[Attempt, ...]
    tokens_reserved: int
    tokens_charged: int
    tokens_estimated: bool
    """True when the provider reported no usable usage and the charge is an estimate."""


class AllProvidersFailedError(Exception):
    def __init__(
        self,
        attempts: Sequence[Attempt],
        *,
        retry_after_seconds: int | None = None,
        all_saturated: bool = False,
        fallback_disabled: bool = False,
    ) -> None:
        super().__init__("all providers failed")
        self.attempts = tuple(attempts)
        self.retry_after_seconds = retry_after_seconds
        self.all_saturated = all_saturated
        self.fallback_disabled = fallback_disabled


class UpstreamRejectedError(Exception):
    """A provider answered with a client error; failing over would not change the outcome."""

    def __init__(self, status: int, message: str, attempts: Sequence[Attempt]) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.attempts = tuple(attempts)


class UnsupportedByProvidersError(Exception):
    """No provider dialect can express this request; the caller must change it."""

    def __init__(self, message: str, attempts: Sequence[Attempt]) -> None:
        super().__init__(message)
        self.message = message
        self.attempts = tuple(attempts)


def sanitized_provider_message(response: ProviderResponse | None) -> str:
    """A short, single-line provider message, or a generic one when unsuitable.

    The provider's text is relayed only when it is plain prose: a conservative
    character allow-list keeps URLs, file paths, e-mail addresses and anything
    resembling a credential or stack trace out of client-facing payloads.
    """
    fallback = "The model provider rejected the request"
    if response is None or not isinstance(response.body, dict):
        return fallback
    error = response.body.get("error")
    message: Any = None
    if isinstance(error, dict):
        message = error.get("message")
    elif isinstance(error, str):
        message = error
    else:
        message = response.body.get("message")
    if not isinstance(message, str):
        return fallback
    collapsed = " ".join(message.split())
    if not collapsed or not _SAFE_MESSAGE_RE.match(collapsed):
        return fallback
    lowered = collapsed.lower()
    if "traceback" in lowered or " file " in lowered:
        return fallback
    if any(len(token) > _MAX_MESSAGE_TOKEN_LENGTH for token in collapsed.split()):
        return fallback
    return collapsed


class ModelRouter:
    def __init__(
        self,
        providers: Sequence[ProviderClient],
        limiter: SlidingWindowTokenLimiter,
        *,
        estimator: TokenEstimator | None = None,
        calibration: Calibration | None = None,
        default_max_tokens: int = 1024,
        failover_statuses: Sequence[int] = (408, 429, 500, 502, 503, 504, 529),
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 30.0,
        max_cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self._providers = tuple(providers)
        self._limiter = limiter
        self._estimator = estimator or HeuristicTokenEstimator()
        self.calibration = calibration or Calibration()
        self._default_max_tokens = default_max_tokens
        self._failover_statuses = frozenset(failover_statuses) | {429}
        self._max_cooldown = max_cooldown_seconds
        self._breakers = {
            provider.name: CircuitBreaker(
                failure_threshold=circuit_failure_threshold,
                recovery_seconds=circuit_recovery_seconds,
                clock=clock,
            )
            for provider in self._providers
        }

    @property
    def providers(self) -> tuple[ProviderClient, ...]:
        return self._providers

    def breaker(self, provider_name: str) -> CircuitBreaker:
        return self._breakers[provider_name]

    # -- estimation ------------------------------------------------------------

    def estimate_prompt(self, payload: dict[str, Any]) -> int:
        return prompt_tokens(payload, self._estimator)

    def estimate_reservation(self, tenant_id: str, payload: dict[str, Any]) -> int:
        """Prompt estimate scaled by the tenant's worst calibration factor, plus the budget."""
        factor = self.calibration.worst_factor(tenant_id, (p.name for p in self._providers))
        prompt = math.ceil(self.estimate_prompt(payload) * factor)
        completion = requested_completion_tokens(payload, self._default_max_tokens)
        return prompt + completion * requested_choices(payload)

    # -- routing ---------------------------------------------------------------

    async def complete(
        self,
        tenant_id: str,
        payload: dict[str, Any],
        request_id: str,
        *,
        tenant_model: str | None = None,
        allow_fallback: bool = True,
        client_headers: Mapping[str, str] | None = None,
    ) -> RoutingResult:
        """Route one completion request.

        Raises ``RateLimitExceededError``, ``UpstreamRejectedError``,
        ``UnsupportedByProvidersError`` or ``AllProvidersFailedError``.
        """
        prompt_estimate = self.estimate_prompt(payload)
        # The ledger key is minted here, never taken from the caller: a reused
        # or shared X-Request-Id must not collide in the UNIQUE column.
        reservation = await self._limiter.reserve(
            tenant_id, uuid.uuid4().hex, self.estimate_reservation(tenant_id, payload)
        )
        attempts: list[Attempt] = []
        settled = False
        unsupported: list[str] = []
        retry_after: float | None = None
        try:
            for position, provider in enumerate(self._providers):
                if position > 0 and not allow_fallback:
                    attempts.append(Attempt(provider.name, "fallback_disabled"))
                    self._log_attempt(request_id, attempts[-1])
                    break
                breaker = self._breakers[provider.name]
                if not breaker.allows_request():
                    attempts.append(Attempt(provider.name, "circuit_open"))
                    self._log_attempt(request_id, attempts[-1])
                    continue
                model = provider.resolve_model(payload, tenant_model)
                try:
                    attempt, response = await self._attempt(
                        provider, payload, request_id, model, client_headers
                    )
                except BaseException:
                    # Cancelled or crashed before a verdict: a half-open trial must
                    # not stay marked in flight forever.
                    breaker.abandon_trial()
                    raise
                attempts.append(attempt)
                self._log_attempt(request_id, attempt)

                if attempt.outcome == "success":
                    breaker.record_success()
                    assert response is not None
                    assert isinstance(response.body, dict)
                    # From here on the tenant consumed provider capacity: whatever happens
                    # to this task (cancellation included), the reservation is never released.
                    settled = True
                    charged, estimated = await self._settle(
                        reservation,
                        tenant_id,
                        provider.name,
                        prompt_estimate,
                        response.body,
                        request_id,
                    )
                    return RoutingResult(
                        body=response.body,
                        provider=provider.name,
                        model=model,
                        fallback=position > 0,
                        attempts=tuple(attempts),
                        tokens_reserved=reservation.tokens,
                        tokens_charged=charged,
                        tokens_estimated=estimated,
                    )
                if attempt.outcome == "rejected":
                    # The provider is healthy; the request itself is unacceptable.
                    breaker.record_success()
                    assert response is not None
                    raise UpstreamRejectedError(
                        response.status, sanitized_provider_message(response), attempts
                    )
                if attempt.outcome == "unsupported":
                    unsupported.append(_unsupported_message(response))
                    continue
                if attempt.outcome in _LOCAL_OUTCOMES:
                    continue  # saturation is local capacity, not provider health
                if response is not None:
                    hint = parse_retry_after(response.retry_after, cap_seconds=self._max_cooldown)
                    if hint is not None:
                        breaker.cool_down(hint)
                        retry_after = max(retry_after or 0.0, hint)
                breaker.record_failure()

            if unsupported and len(unsupported) == len(
                [a for a in attempts if a.outcome not in ("circuit_open", "fallback_disabled")]
            ):
                raise UnsupportedByProvidersError(unsupported[0], attempts)
            raise AllProvidersFailedError(
                attempts,
                retry_after_seconds=math.ceil(retry_after) if retry_after else None,
                all_saturated=bool(attempts) and all(a.outcome == "saturated" for a in attempts),
                fallback_disabled=any(a.outcome == "fallback_disabled" for a in attempts),
            )
        finally:
            if not settled:
                await self._release_quietly(reservation)

    async def _settle(
        self,
        reservation: Reservation,
        tenant_id: str,
        provider_name: str,
        prompt_estimate: int,
        body: dict[str, Any],
        request_id: str,
    ) -> tuple[int, bool]:
        """Charge the real usage, or an estimate of it, and feed the calibration."""
        usage = reported_usage(body)
        estimated = usage is None
        if usage is not None:
            charged = usage.total
            if usage.prompt is not None:
                self.calibration.observe(tenant_id, provider_name, prompt_estimate, usage.prompt)
        else:
            factor = self.calibration.factor(tenant_id, provider_name)
            generated = completion_tokens_from_body(body, self._estimator)
            charged = math.ceil((prompt_estimate + generated) * factor)
        try:
            await self._limiter.settle(reservation, charged)
        except Exception:
            # The provider already answered; losing that answer over a
            # bookkeeping failure would be the worse outcome. The
            # estimated reservation simply stays charged.
            logger.exception(
                "settlement failed; the reservation stays charged",
                extra={"event": "router.settle_failed", "request_id": request_id},
            )
            charged = reservation.tokens
        return charged, estimated

    async def _release_quietly(self, reservation: Reservation) -> None:
        try:
            await self._limiter.release(reservation)
        except Exception:
            logger.exception(
                "failed to release reservation",
                extra={"event": "router.release_failed", "ledger_id": reservation.request_id},
            )

    async def _attempt(
        self,
        provider: ProviderClient,
        payload: dict[str, Any],
        request_id: str,
        model: str | None,
        client_headers: Mapping[str, str] | None,
    ) -> tuple[Attempt, ProviderResponse | None]:
        started = time.perf_counter()
        try:
            response = await provider.complete(
                payload,
                request_id,
                model=model,
                default_max_tokens=self._default_max_tokens,
                client_headers=client_headers,
            )
        except UnsupportedRequestError as exc:
            return Attempt(provider.name, "unsupported", None, 0.0, model), _detail(exc)
        except ProviderSaturatedError:
            return Attempt(provider.name, "saturated", None, 0.0, model), None
        except ProviderTimeoutError:
            return Attempt(provider.name, "timeout", None, _ms_since(started), model), None
        except ProviderUnavailableError:
            return Attempt(provider.name, "unavailable", None, _ms_since(started), model), None

        status = response.status
        latency = response.latency_ms
        if 200 <= status < 300:
            if isinstance(response.body, dict):
                return Attempt(provider.name, "success", status, latency, model), response
            return Attempt(provider.name, "unparseable", status, latency, model), response
        if status == 429:
            return Attempt(provider.name, "rate_limited", status, latency, model), response
        if status in self._failover_statuses or status in provider.failover_statuses:
            return Attempt(provider.name, "server_error", status, latency, model), response
        if status >= 500:
            return Attempt(provider.name, "server_error", status, latency, model), response
        if 400 <= status < 500:
            return Attempt(provider.name, "rejected", status, latency, model), response
        return Attempt(provider.name, "server_error", status, latency, model), response

    @staticmethod
    def _log_attempt(request_id: str, attempt: Attempt) -> None:
        level = logging.INFO if attempt.outcome == "success" else logging.WARNING
        logger.log(
            level,
            "provider attempt",
            extra={"event": "router.attempt", "request_id": request_id, **attempt.as_dict()},
        )


def _detail(exc: UnsupportedRequestError) -> ProviderResponse:
    """Carry the dialect's explanation through the attempt tuple without a real response."""
    return ProviderResponse(
        status=0, body={"error": {"message": str(exc)}}, raw_length=0, latency_ms=0.0
    )


def _unsupported_message(response: ProviderResponse | None) -> str:
    if response is not None and isinstance(response.body, dict):
        error = response.body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])
    return "the request cannot be expressed in the provider's dialect"


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
