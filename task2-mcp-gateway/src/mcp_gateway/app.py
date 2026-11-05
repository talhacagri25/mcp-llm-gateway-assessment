"""Starlette application: the request pipeline of the gateway.

Pipeline for ``POST <mcp_path>``:

1. assign or validate a request id,
2. authenticate the bearer token (401 + ``WWW-Authenticate`` on failure),
3. read the body under a size limit (413 on overflow),
4. parse the JSON-RPC payload (400 with ``-32700`` / ``-32600`` on failure),
5. evaluate the tool access policy for every message,
6. either answer locally with ``-32001 Unauthorized Tool Call`` or relay the
   untouched body to the downstream and stream its response back.

``GET`` and ``DELETE`` on the MCP path (used by the Streamable HTTP transport
for server-initiated streams and session termination) are authenticated and
relayed transparently.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from mcp_gateway.auth import (
    AuthenticationError,
    Principal,
    StaticTokenVerifier,
    TokenVerifier,
    authenticate,
)
from mcp_gateway.config import GatewaySettings
from mcp_gateway.jsonrpc import (
    DOWNSTREAM_TIMEOUT,
    DOWNSTREAM_UNAVAILABLE,
    GATEWAY_OVERLOADED,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    UNAUTHENTICATED,
    UNAUTHORIZED_TOOL_CALL,
    JsonRpcMessage,
    JsonRpcPayload,
    JsonRpcProtocolError,
    error_response,
    parse_payload,
)
from mcp_gateway.logging_config import configure_logging
from mcp_gateway.policy import Decision, ToolAccessPolicy, Verdict
from mcp_gateway.proxy import (
    HEADER_REQUEST_ID,
    DownstreamClient,
    DownstreamError,
    build_forward_headers,
    filter_response_headers,
)

logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class BodyTooLargeError(Exception):
    pass


class RelayInterruptedError(Exception):
    """The downstream failed after its headers were relayed; already logged at the source."""


class RelayedResponse(StreamingResponse):
    """A streamed downstream body whose clean-up runs however the response ends.

    ``on_close`` releases the caller's concurrency slot and closes the upstream
    connection. It runs after a complete body, after a failure in the middle
    of it, and also when the response could not even be started (the client
    went away before the status line was written), which a background task
    or a ``finally`` inside the body iterator would both miss.
    """

    def __init__(self, *args: Any, on_close: Callable[[], Awaitable[None]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._on_close()


class JsonRpcResponse(JSONResponse):
    """JSON-RPC bodies that can echo caller supplied values safely.

    ``ensure_ascii`` keeps lone surrogates (which JSON escapes can express) as
    escapes instead of failing to encode, and ``allow_nan`` refuses to emit
    tokens no JSON parser accepts.
    """

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode("ascii")


class InflightSlots:
    """Per-subject count of requests currently being served (including open streams)."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._counts: dict[str, int] = {}

    def acquire(self, subject: str) -> Callable[[], None] | None:
        """Take a slot and return an idempotent release function, or ``None`` when full."""
        current = self._counts.get(subject, 0)
        if self._limit and current >= self._limit:
            return None
        self._counts[subject] = current + 1
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            remaining = self._counts.get(subject, 0) - 1
            if remaining > 0:
                self._counts[subject] = remaining
            else:
                self._counts.pop(subject, None)

        return release

    def count(self, subject: str) -> int:
        return self._counts.get(subject, 0)


def resolve_request_id(candidate: str | None) -> str:
    """Honour a well formed client supplied request id, otherwise mint one."""
    if candidate and _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


async def read_body_limited(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise BodyTooLargeError
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise BodyTooLargeError
        chunks.append(chunk)
    return b"".join(chunks)


class GatewayService:
    def __init__(
        self,
        settings: GatewaySettings,
        verifier: TokenVerifier,
        policy: ToolAccessPolicy,
        downstream: DownstreamClient,
    ) -> None:
        self._settings = settings
        self._verifier = verifier
        self._policy = policy
        self._downstream = downstream
        self.inflight = InflightSlots(settings.max_inflight_per_subject)
        # GET opens the long-lived server-to-client event stream; it gets its own
        # idle bound instead of the request/response read timeout.
        self._stream_timeout = httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.stream_read_timeout_seconds,
            write=10.0,
            pool=5.0,
        )

    # -- responses -----------------------------------------------------------

    @staticmethod
    def _json(
        status: int, body: Any, request_id: str, extra_headers: dict[str, str] | None = None
    ) -> Response:
        headers = {HEADER_REQUEST_ID: request_id, **(extra_headers or {})}
        return JsonRpcResponse(body, status_code=status, headers=headers)

    # -- pipeline ------------------------------------------------------------

    async def handle_mcp(self, request: Request) -> Response:
        started = time.perf_counter()
        request_id = resolve_request_id(request.headers.get(HEADER_REQUEST_ID))

        try:
            principal = authenticate(request.headers.get("authorization"), self._verifier)
        except AuthenticationError as exc:
            logger.warning(
                "authentication failed",
                extra={"event": "auth.denied", "request_id": request_id, "reason": exc.reason},
            )
            return self._json(
                401,
                error_response(
                    None,
                    UNAUTHENTICATED,
                    f"Unauthenticated: {exc.description}",
                    {"reason": exc.reason},
                ),
                request_id,
                {"WWW-Authenticate": exc.www_authenticate},
            )

        release = self.inflight.acquire(principal.subject)
        if release is None:
            logger.warning(
                "per-subject concurrency cap reached",
                extra={
                    "event": "request.throttled",
                    "request_id": request_id,
                    "subject": principal.subject,
                    "limit": self._settings.max_inflight_per_subject,
                },
            )
            return self._json(
                429,
                error_response(
                    None,
                    GATEWAY_OVERLOADED,
                    "Too many concurrent requests for this caller",
                    {"limit": self._settings.max_inflight_per_subject},
                ),
                request_id,
                {"Retry-After": "1"},
            )
        try:
            response = await self._authenticated(request, principal, request_id, started, release)
        except BaseException:
            release()
            raise
        if not isinstance(response, StreamingResponse):
            release()  # relayed streams release when their body completes
        return response

    async def _authenticated(
        self,
        request: Request,
        principal: Principal,
        request_id: str,
        started: float,
        release: Callable[[], None],
    ) -> Response:
        if request.method != "POST":
            return await self._relay(request, principal, request_id, None, None, started, release)

        try:
            body = await read_body_limited(request, self._settings.max_body_bytes)
        except ClientDisconnect:
            logger.warning(
                "client disconnected while uploading",
                extra={
                    "event": "client.disconnected",
                    "request_id": request_id,
                    "subject": principal.subject,
                },
            )
            return self._json(
                400, error_response(None, INVALID_REQUEST, "Client went away"), request_id
            )
        except BodyTooLargeError:
            logger.warning(
                "request body too large",
                extra={
                    "event": "request.too_large",
                    "request_id": request_id,
                    "subject": principal.subject,
                },
            )
            return self._json(
                413,
                error_response(
                    None,
                    INVALID_REQUEST,
                    f"Request body exceeds {self._settings.max_body_bytes} bytes",
                ),
                request_id,
            )

        try:
            payload = parse_payload(body, max_batch_size=self._settings.max_batch_size)
        except JsonRpcProtocolError as exc:
            logger.warning(
                "malformed JSON-RPC payload",
                extra={
                    "event": "request.malformed",
                    "request_id": request_id,
                    "subject": principal.subject,
                    "code": exc.code,
                },
            )
            return self._json(
                400, error_response(exc.request_id, exc.code, exc.message, exc.data), request_id
            )

        decisions = [
            (message, self._policy.evaluate(principal, message)) for message in payload.messages
        ]
        for message, decision in decisions:
            self._audit(request_id, principal, message, decision, payload.is_batch)

        if any(decision.verdict is not Verdict.ALLOW for _, decision in decisions):
            return self._reject(payload, decisions, principal, request_id)

        return await self._relay(request, principal, request_id, body, payload, started, release)

    # -- authorisation outcomes ---------------------------------------------

    def _audit(
        self,
        request_id: str,
        principal: Principal,
        message: JsonRpcMessage,
        decision: Decision,
        is_batch: bool,
    ) -> None:
        level = logging.INFO if decision.verdict is Verdict.ALLOW else logging.WARNING
        logger.log(
            level,
            "authorization decision",
            extra={
                "event": "authz.decision",
                "request_id": request_id,
                "subject": principal.subject,
                "role": principal.role,
                "token_fp": principal.token_fingerprint,
                "method": message.method,
                "tool": decision.tool,
                "verdict": decision.verdict.value,
                "reason": decision.reason,
                "batch": is_batch,
            },
        )

    def _error_for(
        self,
        message: JsonRpcMessage,
        decision: Decision,
        principal: Principal,
        *,
        batch_denied: bool,
    ) -> dict[str, Any]:
        if decision.verdict is Verdict.DENY:
            return error_response(
                message.request_id,
                UNAUTHORIZED_TOOL_CALL,
                "Unauthorized Tool Call",
                {
                    "reason": decision.reason,
                    "tool": decision.tool,
                    "required_roles": list(decision.required_roles),
                    "role": principal.role,
                    "detail": decision.message,
                },
            )
        if decision.verdict is Verdict.INVALID:
            return error_response(
                message.request_id, INVALID_PARAMS, decision.message, {"reason": decision.reason}
            )
        # An allowed request inside a rejected batch: say why its batch was refused.
        data: dict[str, Any] = {}
        if decision.tool is not None:
            data["tool"] = decision.tool
        if batch_denied:
            data["reason"] = "batch_contains_unauthorized_tool_call"
            return error_response(
                message.request_id,
                UNAUTHORIZED_TOOL_CALL,
                "Unauthorized Tool Call: batch rejected",
                data,
            )
        data["reason"] = "batch_contains_invalid_request"
        return error_response(
            message.request_id,
            INVALID_REQUEST,
            "Invalid Request: batch rejected because another element is invalid",
            data,
        )

    def _reject(
        self,
        payload: JsonRpcPayload,
        decisions: list[tuple[JsonRpcMessage, Decision]],
        principal: Principal,
        request_id: str,
    ) -> Response:
        any_invalid = any(d.verdict is Verdict.INVALID for _, d in decisions)
        any_denied = any(d.verdict is Verdict.DENY for _, d in decisions)
        status = 400 if any_invalid else self._settings.denied_http_status

        errors = [
            self._error_for(message, decision, principal, batch_denied=any_denied)
            for message, decision in decisions
            if message.is_request
        ]

        if payload.is_batch:
            body: Any = errors
            if not errors:
                # Only notifications were present: there is no id to answer, so a
                # single error with a null id and the HTTP status carry the verdict.
                body = error_response(None, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call")
                status = 400 if any_invalid else 403
            return self._json(status, body, request_id)

        message, decision = decisions[0]
        if message.is_request:
            return self._json(status, errors[0], request_id)
        # A notification has no reply channel in JSON-RPC; the HTTP status is the signal.
        body = self._error_for(message, decision, principal, batch_denied=any_denied)
        body["id"] = None
        return self._json(400 if any_invalid else 403, body, request_id)

    # -- relaying ------------------------------------------------------------

    async def _relay(
        self,
        request: Request,
        principal: Principal,
        request_id: str,
        body: bytes | None,
        payload: JsonRpcPayload | None,
        started: float,
        release: Callable[[], None],
    ) -> Response:
        headers = build_forward_headers(
            request.headers,
            principal=principal,
            request_id=request_id,
            client_host=request.client.host if request.client else None,
            service_token=self._settings.downstream_service_token,
        )
        timeouts = self._stream_timeout if request.method == "GET" else None
        try:
            upstream = await self._downstream.send(
                request.method, headers, body, request.url.query, timeouts=timeouts
            )
        except DownstreamError as exc:
            logger.error(
                "downstream request failed",
                extra={
                    "event": "proxy.downstream_error",
                    "request_id": request_id,
                    "kind": exc.kind,
                    "detail": exc.detail,
                },
            )
            return self._downstream_failure(exc, payload, request_id)

        logger.info(
            "request relayed",
            extra={
                "event": "proxy.relayed",
                "request_id": request_id,
                "subject": principal.subject,
                "http_method": request.method,
                "downstream_status": upstream.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )

        async def finished() -> None:
            try:
                await upstream.aclose()
            finally:
                release()

        response = RelayedResponse(
            self._relay_body(upstream, request_id, release),
            status_code=upstream.status_code,
            on_close=finished,
        )
        request_id_header = HEADER_REQUEST_ID.lower().encode("latin-1")
        relayed = [(k, v) for k, v in filter_response_headers(upstream) if k != request_id_header]
        relayed.append((request_id_header, request_id.encode("latin-1")))
        response.raw_headers = relayed
        return response

    @staticmethod
    async def _relay_body(
        upstream: httpx.Response, request_id: str, release: Callable[[], None]
    ) -> AsyncIterator[bytes]:
        """Stream the downstream body; a failure mid-way is logged, then the connection drops.

        Once the status line and headers are out there is no clean way to report
        an error, so the failure is re-raised: the server closes the connection
        without the final chunk and the client sees a truncated response instead
        of a silently complete one.
        """
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.HTTPError as exc:
            logger.error(
                "downstream stream interrupted",
                extra={
                    "event": "proxy.stream_interrupted",
                    "request_id": request_id,
                    "kind": "timeout" if isinstance(exc, httpx.TimeoutException) else "unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
            raise RelayInterruptedError(str(exc)) from exc
        finally:
            release()

    def _downstream_failure(
        self, exc: DownstreamError, payload: JsonRpcPayload | None, request_id: str
    ) -> Response:
        headers: dict[str, str] = {}
        if exc.kind == "timeout":
            status, code, message = 504, DOWNSTREAM_TIMEOUT, "Downstream MCP server timed out"
        elif exc.kind == "overloaded":
            status, code = 503, GATEWAY_OVERLOADED
            message = "Gateway connection pool exhausted; retry shortly"
            headers["Retry-After"] = "1"
        else:
            status, code, message = 502, DOWNSTREAM_UNAVAILABLE, "Downstream MCP server unavailable"
        data = {"request_id": request_id}
        if payload is None or not payload.requests:
            body: Any = error_response(None, code, message, data)
        elif payload.is_batch:
            body = [error_response(m.request_id, code, message, data) for m in payload.requests]
        else:
            body = error_response(payload.requests[0].request_id, code, message, data)
        return self._json(status, body, request_id, headers)

    # -- misc endpoints ------------------------------------------------------

    async def health(self, request: Request) -> Response:
        """Liveness only; nothing about the deployment is disclosed to anonymous callers."""
        return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

    async def http_error(self, request: Request, exc: Exception) -> Response:
        """Routing errors (unknown path, wrong method) as JSON-RPC errors."""
        assert isinstance(exc, HTTPException)
        request_id = resolve_request_id(request.headers.get(HEADER_REQUEST_ID))
        return self._json(
            exc.status_code,
            error_response(None, INVALID_REQUEST, f"Invalid Request: {exc.detail}"),
            request_id,
            dict(exc.headers or {}),
        )

    async def unhandled_error(self, request: Request, exc: Exception) -> Response:
        request_id = resolve_request_id(request.headers.get(HEADER_REQUEST_ID))
        if not isinstance(exc, RelayInterruptedError):
            # Relay interruptions were logged where they happened; everything else is news.
            logger.exception(
                "unhandled error", extra={"event": "gateway.error", "request_id": request_id}
            )
        return self._json(
            500, error_response(None, INTERNAL_ERROR, "Internal gateway error"), request_id
        )


def _load_verifier(settings: GatewaySettings) -> TokenVerifier:
    allow = settings.allow_development_credentials
    if settings.tokens_json is not None:
        return StaticTokenVerifier.from_json(settings.tokens_json, allow_development=allow)
    assert settings.tokens_file is not None  # guaranteed by GatewaySettings
    return StaticTokenVerifier.from_json_file(settings.tokens_file, allow_development=allow)


def create_app(
    settings: GatewaySettings,
    *,
    verifier: TokenVerifier | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Starlette:
    """Build the gateway application.

    ``verifier`` and ``http_client`` are injectable for tests; in production
    both are derived from ``settings``.
    """
    verifier = verifier or _load_verifier(settings)
    policy = ToolAccessPolicy.default(settings.admin_tool_prefixes, settings.admin_roles)
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.read_timeout_seconds,
            write=10.0,
            pool=5.0,
        ),
        limits=httpx.Limits(
            max_connections=settings.max_connections,
            max_keepalive_connections=min(settings.max_connections, 100),
        ),
        follow_redirects=False,
    )
    service = GatewayService(
        settings, verifier, policy, DownstreamClient(client, settings.downstream_url)
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        logger.info(
            "gateway ready",
            extra={
                "event": "gateway.start",
                "mcp_path": settings.mcp_path,
                "admin_tool_prefixes": list(settings.admin_tool_prefixes),
                "admin_roles": list(settings.admin_roles),
            },
        )
        try:
            yield
        finally:
            if owns_client:
                await client.aclose()

    app = Starlette(
        routes=[
            Route(settings.mcp_path, service.handle_mcp, methods=["POST", "GET", "DELETE"]),
            Route("/healthz", service.health, methods=["GET"]),
        ],
        lifespan=lifespan,
        exception_handlers={HTTPException: service.http_error, Exception: service.unhandled_error},
    )
    # No unauthenticated 307 to "/mcp/" whose Location is built from the client's Host.
    app.router.redirect_slashes = False
    app.state.service = service  # inspectable by tests and operators' debug tooling
    return app


def app_from_env() -> Starlette:
    """Application factory for ``uvicorn --factory mcp_gateway.app:app_from_env``."""
    settings = GatewaySettings.from_env()
    configure_logging(settings.log_level)
    return create_app(settings)
