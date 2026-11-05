"""HTTP surface of the router: ``POST /v1/chat/completions``.

Every error leaves the gateway in the same payload shape:

    {"error": {"type": "...", "message": "...", "request_id": "..."}}

with no provider URLs, stack traces or exception text. Rate limit state is
exposed through ``X-RateLimit-*`` headers and ``Retry-After``; which provider
and model served the request, and whether that was a fallback, through
``X-Model-Router-*`` headers.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from model_router.config import (
    MODEL_NAME_RULE,
    ConfigurationError,
    RouterSettings,
    TenantSettings,
    valid_model_name,
)
from model_router.logging_config import configure_logging
from model_router.providers import ProviderClient
from model_router.rate_limiter import (
    LedgerUnavailableError,
    RateLimitExceededError,
    RateLimitSnapshot,
    SlidingWindowTokenLimiter,
)
from model_router.router import (
    LOCAL_OUTCOMES,
    AllProvidersFailedError,
    Attempt,
    ModelRouter,
    RoutingResult,
    UnsupportedByProvidersError,
    UpstreamRejectedError,
)
from model_router.storage import TokenLedgerStore
from model_router.token_estimator import Calibration, load_estimator

logger = logging.getLogger(__name__)

HEADER_REQUEST_ID = "X-Request-Id"
HEADER_PROVIDER = "X-Model-Router-Provider"
HEADER_MODEL = "X-Model-Router-Model"
HEADER_FALLBACK = "X-Model-Router-Fallback"
HEADER_TOKENS_ESTIMATED = "X-Model-Router-Tokens-Estimated"
HEADER_ALLOW_FALLBACK = "X-Model-Router-Allow-Fallback"
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_RELAYED_CLIENT_ERRORS = frozenset({400, 404, 413, 422})
MAX_MESSAGES = 4096
"""Longest conversation accepted; beyond this the per-message work stops being bounded by
the body size limit alone."""
MAX_CONTENT_PARTS = 4096
MAX_RECORDED_ATTEMPTS = 16
READINESS_PROBE_INTERVAL = 1.0
"""Seconds a readiness verdict is reused, so probes cannot be turned into ledger lock churn."""
_HTTP_ERROR_TYPES = {404: "not_found", 405: "method_not_allowed"}
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class SafeJsonResponse(JSONResponse):
    """JSON that cannot fail to render on odd provider output (lone surrogates)."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8", errors="replace")


class GatewayError(Exception):
    def __init__(
        self,
        status: int,
        error_type: str,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message
        self.headers = headers or {}


def resolve_request_id(candidate: str | None) -> str:
    if candidate and _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def error_payload(error_type: str, message: str, request_id: str, **extra: Any) -> dict[str, Any]:
    error: dict[str, Any] = {"type": error_type, "message": message, "request_id": request_id}
    error.update({k: v for k, v in extra.items() if v is not None})
    return {"error": error}


def rate_limit_headers(snapshot: RateLimitSnapshot) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(snapshot.limit),
        "X-RateLimit-Remaining": str(snapshot.remaining),
        "X-RateLimit-Window-Seconds": str(snapshot.window_seconds),
    }


def _positive_int(payload: dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GatewayError(400, "invalid_request", f"'{key}' must be a positive integer")


def validate_completion_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GatewayError(400, "invalid_request", "Request body must be a JSON object")
    stream = payload.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise GatewayError(400, "invalid_request", "'stream' must be a boolean")
    if stream is True:
        raise GatewayError(400, "invalid_request", "Streaming is not supported by this endpoint")
    model = payload.get("model")
    if model is not None and not valid_model_name(model):
        raise GatewayError(400, "invalid_request", f"'model' must be {MODEL_NAME_RULE}")
    messages = payload.get("messages")
    prompt = payload.get("prompt")
    if messages is None and prompt is None:
        raise GatewayError(400, "invalid_request", "Provide 'messages' or 'prompt'")
    if messages is not None:
        if not isinstance(messages, list) or not messages:
            raise GatewayError(400, "invalid_request", "'messages' must be a non-empty array")
        if len(messages) > MAX_MESSAGES:
            raise GatewayError(
                400, "invalid_request", f"'messages' may hold at most {MAX_MESSAGES} entries"
            )
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise GatewayError(400, "invalid_request", "Each message needs a string 'role'")
            content = message.get("content")
            if content is not None and not isinstance(content, str | list):
                raise GatewayError(
                    400, "invalid_request", "'content' must be a string or an array of parts"
                )
            if isinstance(content, list):
                if len(content) > MAX_CONTENT_PARTS:
                    raise GatewayError(
                        400,
                        "invalid_request",
                        f"'content' may hold at most {MAX_CONTENT_PARTS} parts",
                    )
                if not all(isinstance(p, dict) for p in content):
                    raise GatewayError(
                        400, "invalid_request", "Each content part must be an object"
                    )
    if prompt is not None and not isinstance(prompt, str):
        raise GatewayError(400, "invalid_request", "'prompt' must be a string")
    for key in ("max_tokens", "max_completion_tokens", "n"):
        _positive_int(payload, key)
    return payload


def parse_allow_fallback(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise GatewayError(400, "invalid_request", f"{HEADER_ALLOW_FALLBACK} must be true or false")


class RouterService:
    def __init__(
        self,
        settings: RouterSettings,
        store: TokenLedgerStore,
        limiter: SlidingWindowTokenLimiter,
        router: ModelRouter,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._store = store
        self._limiter = limiter
        self._router = router
        self._clock = clock
        self._tenants = {key.encode("utf-8"): tenant for key, tenant in settings.tenants.items()}
        self._probe_lock = asyncio.Lock()
        self._probe_checked_at = float("-inf")
        self._probe_ok = True

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _json(
        status: int, body: Any, request_id: str, headers: dict[str, str] | None = None
    ) -> Response:
        return SafeJsonResponse(
            body,
            status_code=status,
            headers={HEADER_REQUEST_ID: request_id, "Cache-Control": "no-store", **(headers or {})},
        )

    def _authenticate(self, request: Request) -> TenantSettings:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        presented = token.strip().encode("utf-8")
        if scheme.lower() != "bearer" or not presented:
            raise GatewayError(
                401,
                "authentication_error",
                "A tenant API key is required",
                {"WWW-Authenticate": 'Bearer realm="model-router"'},
            )
        match: TenantSettings | None = None
        for key, tenant in self._tenants.items():
            if hmac.compare_digest(key, presented):
                match = tenant
        if match is None:
            raise GatewayError(
                401,
                "authentication_error",
                "Invalid tenant API key",
                {"WWW-Authenticate": 'Bearer realm="model-router", error="invalid_token"'},
            )
        return match

    async def _read_json(self, request: Request) -> Any:
        limit = self._settings.max_body_bytes
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise GatewayError(413, "request_too_large", f"Request body exceeds {limit} bytes")
        chunks: list[bytes] = []
        total = 0
        try:
            async for chunk in request.stream():
                total += len(chunk)
                if total > limit:
                    raise GatewayError(
                        413, "request_too_large", f"Request body exceeds {limit} bytes"
                    )
                chunks.append(chunk)
        except ClientDisconnect as exc:
            raise GatewayError(400, "client_disconnected", "Client went away") from exc
        try:
            return json.loads(b"".join(chunks))
        except (ValueError, RecursionError) as exc:
            raise GatewayError(400, "invalid_request", "Request body is not valid JSON") from exc

    @staticmethod
    def _check_model_allowed(tenant: TenantSettings, payload: dict[str, Any]) -> None:
        requested = payload.get("model")
        if not tenant.allowed_models:
            return
        if not isinstance(requested, str) or not requested:
            if tenant.model:
                return  # the pin decides; the allow list only bounds explicit requests
            raise GatewayError(
                400, "model_required", "This tenant must name one of its allowed models"
            )
        if requested not in tenant.allowed_models and requested != tenant.model:
            raise GatewayError(
                400, "model_not_allowed", f"Model '{requested}' is not available to this tenant"
            )

    async def _record(
        self,
        request_id: str,
        tenant_id: str,
        outcome: str,
        provider: str | None,
        status: int,
        started: float,
        attempts: Sequence[Attempt],
    ) -> None:
        """Persist one request log row when a provider was actually contacted.

        Admission refusals and other locally decided outcomes are logged but
        not stored: they cost the caller nothing, so a hostile caller could
        otherwise grow the database at will.
        """
        if all(a.outcome in LOCAL_OUTCOMES for a in attempts):
            return
        recorded = [a.as_dict() for a in attempts[:MAX_RECORDED_ATTEMPTS]]
        try:
            await asyncio.to_thread(
                self._store.record_request,
                request_id=request_id,
                tenant_id=tenant_id,
                created_at_ms=int(self._clock() * 1000),
                outcome=outcome,
                provider=provider,
                http_status=status,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                attempts_json=json.dumps(recorded),
            )
        except Exception:
            logger.exception(
                "request log write failed",
                extra={"event": "router.log_failed", "request_id": request_id},
            )

    # -- endpoints -----------------------------------------------------------

    async def chat_completions(self, request: Request) -> Response:
        started = time.perf_counter()
        request_id = resolve_request_id(request.headers.get(HEADER_REQUEST_ID))
        try:
            tenant = self._authenticate(request)
            payload = validate_completion_request(await self._read_json(request))
            self._check_model_allowed(tenant, payload)
            allow_fallback = parse_allow_fallback(
                request.headers.get(HEADER_ALLOW_FALLBACK), self._settings.allow_fallback_default
            )
        except GatewayError as exc:
            logger.warning(
                "request rejected",
                extra={
                    "event": "request.rejected",
                    "request_id": request_id,
                    "status": exc.status,
                    "error_type": exc.error_type,
                },
            )
            return self._json(
                exc.status,
                error_payload(exc.error_type, exc.message, request_id),
                request_id,
                exc.headers,
            )

        tenant_id = tenant.tenant_id
        try:
            result = await self._router.complete(
                tenant_id,
                payload,
                request_id,
                tenant_model=tenant.model,
                allow_fallback=allow_fallback,
                client_headers=request.headers,
            )
        except RateLimitExceededError as exc:
            return await self._rate_limited(exc, request_id, tenant_id, started)
        except LedgerUnavailableError:
            return self._ledger_unavailable(request_id, tenant_id)
        except UpstreamRejectedError as exc:
            return await self._upstream_rejected(exc, request_id, tenant_id, started)
        except UnsupportedByProvidersError as exc:
            logger.warning(
                "request shape unsupported by every provider",
                extra={
                    "event": "router.unsupported",
                    "request_id": request_id,
                    "attempts": [a.as_dict() for a in exc.attempts],
                },
            )
            await self._record(
                request_id, tenant_id, "unsupported", None, 400, started, exc.attempts
            )
            return self._json(
                400,
                error_payload("unsupported_by_provider", exc.message, request_id),
                request_id,
            )
        except AllProvidersFailedError as exc:
            return await self._all_failed(exc, request_id, tenant_id, started)
        return await self._success(result, request_id, tenant_id, started)

    def _ledger_unavailable(self, request_id: str, tenant_id: str) -> Response:
        """Fail closed: without the ledger no admission decision is trustworthy."""
        logger.exception(
            "token ledger unavailable",
            extra={
                "event": "router.ledger_unavailable",
                "request_id": request_id,
                "tenant_id": tenant_id,
            },
        )
        return self._json(
            503,
            error_payload(
                "service_unavailable",
                "The rate limiter storage is temporarily unavailable; please retry shortly",
                request_id,
            ),
            request_id,
            {"Retry-After": "1"},
        )

    async def _all_failed(
        self, exc: AllProvidersFailedError, request_id: str, tenant_id: str, started: float
    ) -> Response:
        logger.error(
            "all providers failed",
            extra={
                "event": "router.failed",
                "request_id": request_id,
                "tenant_id": tenant_id,
                "attempts": [a.as_dict() for a in exc.attempts],
            },
        )
        await self._record(
            request_id, tenant_id, "all_providers_failed", None, 503, started, exc.attempts
        )
        if exc.fallback_disabled:
            message = (
                "The primary provider could not serve the request and fallback was disabled "
                "for this request"
            )
        elif exc.all_saturated:
            message = "All model providers are at capacity; please retry shortly"
        else:
            message = "All model providers are currently unavailable; please retry shortly"
        retry_after = max(exc.retry_after_seconds or 0, 1 if exc.all_saturated else 2)
        return self._json(
            503,
            error_payload("upstream_unavailable", message, request_id),
            request_id,
            {"Retry-After": str(retry_after)},
        )

    async def _success(
        self, result: RoutingResult, request_id: str, tenant_id: str, started: float
    ) -> Response:
        headers = {
            HEADER_PROVIDER: result.provider,
            HEADER_FALLBACK: "true" if result.fallback else "false",
        }
        if result.model:
            headers[HEADER_MODEL] = result.model
        if result.tokens_estimated:
            headers[HEADER_TOKENS_ESTIMATED] = "true"
        window: dict[str, int] = {}
        try:
            snapshot = await self._limiter.snapshot(tenant_id)
        except LedgerUnavailableError:
            # The answer is in hand; missing rate limit headers are the lesser evil.
            logger.exception(
                "window snapshot failed after a served request",
                extra={"event": "router.snapshot_failed", "request_id": request_id},
            )
        else:
            headers.update(rate_limit_headers(snapshot))
            window = {"window_used": snapshot.used, "window_remaining": snapshot.remaining}
        logger.info(
            "completion served",
            extra={
                "event": "router.served",
                "request_id": request_id,
                "tenant_id": tenant_id,
                "provider": result.provider,
                "model": result.model,
                "fallback": result.fallback,
                "tokens_reserved": result.tokens_reserved,
                "tokens_charged": result.tokens_charged,
                "tokens_estimated": result.tokens_estimated,
                "attempts": [a.as_dict() for a in result.attempts],
                **window,
            },
        )
        await self._record(
            request_id, tenant_id, "served", result.provider, 200, started, result.attempts
        )
        return self._json(200, result.body, request_id, headers)

    async def _rate_limited(
        self, exc: RateLimitExceededError, request_id: str, tenant_id: str, started: float
    ) -> Response:
        logger.warning(
            "rate limited",
            extra={
                "event": "router.rate_limited",
                "request_id": request_id,
                "tenant_id": tenant_id,
                "requested": exc.requested,
                "used": exc.snapshot.used,
                "limit": exc.snapshot.limit,
                "retry_after_ms": exc.retry_after_ms,
            },
        )
        headers = rate_limit_headers(exc.snapshot)
        window = exc.snapshot.window_seconds
        if not exc.can_ever_succeed:
            await self._record(request_id, tenant_id, "request_too_large", None, 400, started, ())
            return self._json(
                400,
                error_payload(
                    "request_too_large",
                    f"The request needs about {exc.requested} tokens, more than the "
                    f"{exc.snapshot.limit} tokens per {window} seconds allowed for this tenant",
                    request_id,
                ),
                request_id,
                headers,
            )
        retry_after = exc.retry_after_seconds or 1
        headers["Retry-After"] = str(retry_after)
        await self._record(request_id, tenant_id, "rate_limited", None, 429, started, ())
        return self._json(
            429,
            error_payload(
                "rate_limit_exceeded",
                f"Token budget of {exc.snapshot.limit} tokens per {window} seconds exceeded; "
                f"retry after {retry_after} seconds",
                request_id,
                retry_after_seconds=retry_after,
            ),
            request_id,
            headers,
        )

    async def _upstream_rejected(
        self, exc: UpstreamRejectedError, request_id: str, tenant_id: str, started: float
    ) -> Response:
        attempts = [a.as_dict() for a in exc.attempts]
        if exc.status in (401, 403):
            # A provider credential problem is the gateway's fault, never the caller's.
            logger.error(
                "provider rejected gateway credentials",
                extra={
                    "event": "router.provider_auth_failed",
                    "request_id": request_id,
                    "attempts": attempts,
                },
            )
            await self._record(
                request_id, tenant_id, "provider_auth_failed", None, 502, started, exc.attempts
            )
            return self._json(
                502,
                error_payload(
                    "upstream_unavailable", "The model provider could not be used", request_id
                ),
                request_id,
            )
        status = exc.status if exc.status in _RELAYED_CLIENT_ERRORS else 400
        logger.warning(
            "upstream rejected request",
            extra={
                "event": "router.upstream_rejected",
                "request_id": request_id,
                "status": exc.status,
                "attempts": attempts,
            },
        )
        await self._record(
            request_id, tenant_id, "upstream_rejected", None, status, started, exc.attempts
        )
        return self._json(
            status, error_payload("upstream_rejected", exc.message, request_id), request_id
        )

    async def usage(self, request: Request) -> Response:
        request_id = resolve_request_id(request.headers.get(HEADER_REQUEST_ID))
        try:
            tenant = self._authenticate(request)
        except GatewayError as exc:
            return self._json(
                exc.status,
                error_payload(exc.error_type, exc.message, request_id),
                request_id,
                exc.headers,
            )
        try:
            snapshot = await self._limiter.snapshot(tenant.tenant_id)
        except LedgerUnavailableError:
            return self._ledger_unavailable(request_id, tenant.tenant_id)
        body = {
            "tenant_id": tenant.tenant_id,
            "limit": snapshot.limit,
            "used": snapshot.used,
            "remaining": snapshot.remaining,
            "window_seconds": snapshot.window_seconds,
            "model": tenant.model,
            "allowed_models": list(tenant.allowed_models),
            "calibration": self._router.calibration.factors_for(
                tenant.tenant_id, (p.name for p in self._router.providers)
            ),
        }
        return self._json(200, body, request_id, rate_limit_headers(snapshot))

    async def health(self, request: Request) -> Response:
        """Liveness: the process answers. Nothing about the deployment is disclosed."""
        return SafeJsonResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

    async def readiness(self, request: Request) -> Response:
        """Readiness: the ledger accepts writes. The verdict is cached briefly so
        unauthenticated probes cannot keep taking the ledger's write lock."""
        async with self._probe_lock:
            if time.monotonic() - self._probe_checked_at >= READINESS_PROBE_INTERVAL:
                try:
                    await asyncio.to_thread(self._store.probe)
                except sqlite3.Error:
                    logger.exception(
                        "ledger probe failed", extra={"event": "router.ledger_unhealthy"}
                    )
                    self._probe_ok = False
                else:
                    self._probe_ok = True
                self._probe_checked_at = time.monotonic()
            ready = self._probe_ok
        if ready:
            body: dict[str, str] = {"status": "ok", "ledger": "ok"}
            return SafeJsonResponse(body, headers={"Cache-Control": "no-store"})
        return SafeJsonResponse(
            {"status": "degraded", "ledger": "unavailable"},
            status_code=503,
            headers={"Cache-Control": "no-store", "Retry-After": "1"},
        )

    async def http_error(self, request: Request, exc: Exception) -> Response:
        """Routing errors (unknown path, wrong method) in the standard error shape."""
        assert isinstance(exc, HTTPException)
        request_id = resolve_request_id(request.headers.get(HEADER_REQUEST_ID))
        error_type = _HTTP_ERROR_TYPES.get(exc.status_code, "http_error")
        return self._json(
            exc.status_code,
            error_payload(error_type, exc.detail, request_id),
            request_id,
            dict(exc.headers or {}),
        )

    async def unhandled_error(self, request: Request, exc: Exception) -> Response:
        request_id = resolve_request_id(request.headers.get(HEADER_REQUEST_ID))
        logger.exception(
            "unhandled error", extra={"event": "router.error", "request_id": request_id}
        )
        return self._json(
            500, error_payload("internal_error", "Internal gateway error", request_id), request_id
        )


def build_limiter(
    settings: RouterSettings, store: TokenLedgerStore, *, clock: Callable[[], float] = time.time
) -> SlidingWindowTokenLimiter:
    limiter = SlidingWindowTokenLimiter(
        store,
        tokens_per_window=settings.tokens_per_minute,
        window_seconds=settings.window_seconds,
        clock=clock,
    )
    for tenant in settings.tenants.values():
        if tenant.tokens_per_minute is not None:
            limiter.set_tenant_limit(tenant.tenant_id, tenant.tokens_per_minute)
    return limiter


def build_router(
    settings: RouterSettings,
    limiter: SlidingWindowTokenLimiter,
    http_client: httpx.AsyncClient,
    *,
    clock: Callable[[], float] = time.monotonic,
    store: TokenLedgerStore | None = None,
) -> ModelRouter:
    try:
        estimator = load_estimator(settings.token_estimator)
    except ValueError as exc:
        raise ConfigurationError(f"MODEL_ROUTER_TOKEN_ESTIMATOR: {exc}") from exc
    providers = [
        ProviderClient(
            spec,
            http_client,
            forward_headers=settings.forward_headers,
            max_response_bytes=settings.max_response_bytes,
        )
        for spec in (settings.primary, settings.secondary)
    ]
    return ModelRouter(
        providers,
        limiter,
        estimator=estimator,
        calibration=Calibration(store),
        default_max_tokens=settings.default_max_tokens,
        failover_statuses=settings.failover_statuses,
        circuit_failure_threshold=settings.circuit_failure_threshold,
        circuit_recovery_seconds=settings.circuit_recovery_seconds,
        max_cooldown_seconds=settings.max_cooldown_seconds,
        clock=clock,
    )


def create_app(
    settings: RouterSettings,
    *,
    http_client: httpx.AsyncClient | None = None,
    store: TokenLedgerStore | None = None,
    clock: Callable[[], float] = time.time,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> Starlette:
    """Build the router application; the injectable pieces exist for tests."""
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        follow_redirects=False,
    )
    owns_store = store is None
    ledger = store or TokenLedgerStore(settings.database_path)
    limiter = build_limiter(settings, ledger, clock=clock)
    router = build_router(settings, limiter, client, clock=monotonic_clock, store=ledger)
    service = RouterService(settings, ledger, limiter, router, clock=clock)

    async def evict_forever() -> None:
        while True:
            await asyncio.sleep(settings.eviction_interval_seconds)
            try:
                removed = await limiter.evict_expired()
                logger.debug("eviction pass", extra={"event": "ledger.evicted", "rows": removed})
            except Exception:
                logger.exception("eviction failed", extra={"event": "ledger.eviction_failed"})

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        logger.info(
            "router ready",
            extra={
                "event": "router.start",
                "primary": settings.primary.name,
                "secondary": settings.secondary.name,
                "tokens_per_minute": settings.tokens_per_minute,
                "token_estimator": settings.token_estimator,
            },
        )
        task = asyncio.create_task(evict_forever())
        try:
            yield
        finally:
            task.cancel()
            if owns_client:
                await client.aclose()
            if owns_store:
                ledger.close()

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", service.chat_completions, methods=["POST"]),
            Route("/v1/usage", service.usage, methods=["GET"]),
            Route("/healthz", service.health, methods=["GET"]),
            Route("/readyz", service.readiness, methods=["GET"]),
        ],
        lifespan=lifespan,
        exception_handlers={HTTPException: service.http_error, Exception: service.unhandled_error},
    )
    app.router.redirect_slashes = False  # no 307 built from the client's Host header
    return app


def app_from_env() -> Starlette:
    """Application factory for ``uvicorn --factory model_router.app:app_from_env``."""
    settings = RouterSettings.from_env()
    configure_logging(settings.log_level)
    return create_app(settings)
