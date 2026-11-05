"""Starlette application: the guardrail proxy.

Routes:

* ``POST /v1/chat/completions``: OpenAI Chat Completions dialect
* ``POST /v1/messages``: Anthropic Messages dialect
* ``POST /{v1,v1beta,v1alpha}/models/{model}:generateContent`` and
  ``:streamGenerateContent``: Gemini dialect (streaming with ``?alt=sse``)
* ``GET /healthz``

For streaming requests the upstream SSE byte stream is parsed incrementally,
each event's text deltas pass through a ``StreamingRedactor`` and the event is
re-emitted immediately. Nothing is accumulated beyond one partial SSE event
and the redactor's bounded holdback, so the response starts flowing as soon
as the upstream produces its first token.
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode

import httpx
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_guardrail_gateway import __version__
from llm_guardrail_gateway.adapters import (
    AnthropicMessagesAdapter,
    GeminiGenerateContentAdapter,
    OpenAIChatAdapter,
    RedactorFactory,
    StreamAdapter,
)
from llm_guardrail_gateway.config import GuardrailSettings
from llm_guardrail_gateway.logging_config import configure_logging, redact_url
from llm_guardrail_gateway.redaction.streaming import StreamingRedactor
from llm_guardrail_gateway.sse import SseParseError, SseParser, serialize_event

logger = logging.getLogger(__name__)

HEADER_REQUEST_ID = "X-Request-Id"
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_FORWARDED_REQUEST_HEADERS = frozenset(
    {"accept", "anthropic-version", "anthropic-beta", "openai-organization"}
)
_CLIENT_CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key", "x-goog-api-key"})
_CREDENTIAL_QUERY_PARAMS = frozenset({"key"})
_RELAYED_ERROR_HEADER_PREFIXES = ("retry-after", "x-ratelimit-", "anthropic-ratelimit-")
_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_HTTP_ERROR_TYPES = {404: "not_found", 405: "method_not_allowed"}
_GEMINI_VERSIONS = frozenset({"v1", "v1beta", "v1alpha"})
_GEMINI_ACTIONS = frozenset({"generateContent", "streamGenerateContent"})


class SafeJsonResponse(JSONResponse):
    """JSON that cannot fail to render: lone surrogates are replaced, NaN is refused upstream."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8", errors="replace")


def _reject_constant(name: str) -> Any:
    # NaN and Infinity are not JSON; a provider body carrying them cannot be relayed as JSON.
    raise ValueError(f"non-standard JSON constant {name}")


def load_json_document(raw: bytes | str) -> Any:
    """``json.loads`` for documents the gateway will re-emit: strict about constants."""
    return json.loads(raw, parse_constant=_reject_constant)


class GatewayError(Exception):
    def __init__(self, status: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message


def resolve_request_id(candidate: str | None) -> str:
    if candidate and _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def error_payload(error_type: str, message: str, request_id: str) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": message, "request_id": request_id}}


@dataclass(frozen=True)
class Dialect:
    """What differs between the supported provider wire formats."""

    name: str
    adapter_factory: Callable[[RedactorFactory], StreamAdapter]
    credential_header: str
    """Header carrying the provider credential (``authorization`` means ``Bearer``)."""
    fixed_path: str | None
    """Upstream path, or ``None`` to mirror the request path (Gemini encodes the model in it)."""
    default_headers: dict[str, str] = field(default_factory=dict)

    def credential_headers(self, api_key: str) -> dict[str, str]:
        if self.credential_header == "authorization":
            return {"authorization": f"Bearer {api_key}"}
        return {self.credential_header: api_key}

    def upstream_path(self, request: Request) -> str:
        return self.fixed_path if self.fixed_path is not None else request.url.path

    def wants_stream(self, request: Request, payload: dict[str, Any]) -> bool:
        if self.fixed_path is not None:
            return payload.get("stream") is True
        return request.url.path.endswith(":streamGenerateContent") and (
            request.query_params.get("alt") == "sse"
        )


OPENAI = Dialect("openai", OpenAIChatAdapter, "authorization", "/v1/chat/completions")
ANTHROPIC = Dialect(
    "anthropic",
    AnthropicMessagesAdapter,
    "x-api-key",
    "/v1/messages",
    {"anthropic-version": _DEFAULT_ANTHROPIC_VERSION},
)
GEMINI = Dialect("gemini", GeminiGenerateContentAdapter, "x-goog-api-key", None)


class GuardrailService:
    def __init__(self, settings: GuardrailSettings, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = http_client
        self._patterns = settings.patterns
        self._api_keys = tuple(key.encode("utf-8") for key in settings.api_keys)

    # -- helpers -------------------------------------------------------------

    def new_redactor(self, *, json_mode: bool = False) -> StreamingRedactor:
        return StreamingRedactor(
            self._patterns,
            replacement=self._settings.replacement,
            max_holdback=self._settings.max_holdback_chars,
            json_mode=json_mode,
        )

    @staticmethod
    def _json(
        status: int, body: Any, request_id: str, headers: dict[str, str] | None = None
    ) -> Response:
        return SafeJsonResponse(
            body,
            status_code=status,
            headers={HEADER_REQUEST_ID: request_id, "Cache-Control": "no-store", **(headers or {})},
        )

    @property
    def _bring_your_own_key(self) -> bool:
        """Open gateway without an upstream key: the caller's provider credential is relayed."""
        return not self._api_keys and not self._settings.upstream_api_key

    @staticmethod
    def _presented_credentials(request: Request) -> list[str]:
        """Every place a client SDK may put its key, in order of preference."""
        candidates: list[str] = []
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            candidates.append(token.strip())
        for header in ("x-api-key", "x-goog-api-key"):
            value = request.headers.get(header, "").strip()
            if value:
                candidates.append(value)
        for param in _CREDENTIAL_QUERY_PARAMS:
            value = request.query_params.get(param, "").strip()
            if value:
                candidates.append(value)
        return candidates

    def _authenticate(self, request: Request) -> None:
        """Gateway keys are accepted wherever the provider SDKs put their own key.

        OpenAI clients send ``Authorization: Bearer``, Anthropic clients
        ``x-api-key``, Gemini clients ``x-goog-api-key`` (older ones ``?key=``).
        Forcing one form would make the gateway unusable for the others.
        """
        if not self._api_keys:
            return
        presented = self._presented_credentials(request)
        if not presented:
            raise GatewayError(401, "authentication_error", "A gateway API key is required")
        matched = False
        for candidate in presented:
            encoded = candidate.encode("utf-8")
            for key in self._api_keys:
                matched |= hmac.compare_digest(encoded, key)  # no early exit
        if not matched:
            raise GatewayError(401, "authentication_error", "Invalid gateway API key")

    async def _read_body(self, request: Request) -> bytes:
        limit = self._settings.max_body_bytes
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise GatewayError(413, "request_too_large", f"Request body exceeds {limit} bytes")
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                raise GatewayError(413, "request_too_large", f"Request body exceeds {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    def _upstream_headers(
        self, request: Request, dialect: Dialect, request_id: str
    ) -> dict[str, str]:
        allowed = _FORWARDED_REQUEST_HEADERS | set(self._settings.forward_headers)
        # Lower-case keys so a header can never be sent twice in different spellings.
        headers = {
            name.lower(): value
            for name, value in request.headers.items()
            if name.lower() in allowed
        }
        if self._settings.upstream_api_key:
            headers.update(dialect.credential_headers(self._settings.upstream_api_key))
        elif self._bring_your_own_key:
            # The caller's provider credential is passed through. With gateway keys
            # configured the caller's credential *is* the gateway key and never
            # reaches the provider.
            headers.update(
                {
                    name.lower(): value
                    for name, value in request.headers.items()
                    if name.lower() in _CLIENT_CREDENTIAL_HEADERS
                }
            )
        for name, value in dialect.default_headers.items():
            headers.setdefault(name, value)
        headers["accept-encoding"] = "identity"  # no transparent compression on a token stream
        headers["content-type"] = "application/json"
        headers[HEADER_REQUEST_ID.lower()] = request_id
        headers["user-agent"] = f"llm-guardrail-gateway/{__version__}"
        return headers

    def _upstream_url(self, request: Request, dialect: Dialect) -> str:
        """Base URL plus the dialect path; query parameters relayed minus credentials."""
        pairs = parse_qsl(request.url.query, keep_blank_values=True)
        if not self._bring_your_own_key:
            pairs = [(k, v) for k, v in pairs if k not in _CREDENTIAL_QUERY_PARAMS]
        url = f"{self._settings.upstream_base_url}{dialect.upstream_path(request)}"
        return f"{url}?{urlencode(pairs)}" if pairs else url

    # -- pipeline ------------------------------------------------------------

    async def _handle(self, request: Request, dialect: Dialect) -> Response:
        started = time.perf_counter()
        request_id = resolve_request_id(request.headers.get(HEADER_REQUEST_ID))
        try:
            self._authenticate(request)
            try:
                body = await self._read_body(request)
            except ClientDisconnect as exc:
                raise GatewayError(400, "client_disconnected", "Client went away") from exc
            try:
                payload = json.loads(body)
            except (ValueError, RecursionError) as exc:
                raise GatewayError(
                    400, "invalid_request", "Request body is not valid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise GatewayError(400, "invalid_request", "Request body must be a JSON object")
            wants_stream = dialect.wants_stream(request, payload)

            upstream_request = self._client.build_request(
                "POST",
                self._upstream_url(request, dialect),
                headers=self._upstream_headers(request, dialect, request_id),
                content=body,
            )
            try:
                upstream = await self._client.send(upstream_request, stream=True)
            except httpx.TimeoutException as exc:
                raise GatewayError(504, "upstream_timeout", "The model provider timed out") from exc
            except httpx.TransportError as exc:
                raise GatewayError(
                    502, "upstream_unavailable", "The model provider is unavailable"
                ) from exc
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
                exc.status, error_payload(exc.error_type, exc.message, request_id), request_id
            )

        upstream_ttfb_ms = round((time.perf_counter() - started) * 1000, 2)
        adapter = dialect.adapter_factory(self.new_redactor)
        content_type = upstream.headers.get("content-type", "").lower()
        success = 200 <= upstream.status_code < 300  # SDKs treat every 2xx as an answer
        if success and wants_stream and content_type.startswith("text/event-stream"):
            return self._stream(upstream, adapter, request_id, started, upstream_ttfb_ms, dialect)
        return await self._buffered(upstream, adapter, request_id, started, dialect, success)

    def _stream(
        self,
        upstream: httpx.Response,
        adapter: StreamAdapter,
        request_id: str,
        started: float,
        upstream_ttfb_ms: float,
        dialect: Dialect,
    ) -> Response:
        parser = SseParser(max_event_bytes=self._settings.max_event_bytes)
        deadline = started + self._settings.max_stream_seconds
        byte_budget = self._settings.max_stream_bytes

        async def body() -> AsyncIterator[bytes]:
            first_byte_at: float | None = None
            events_in = 0
            bytes_out = 0
            outcome = "complete"
            try:
                async for chunk in upstream.aiter_bytes():
                    if time.perf_counter() > deadline or bytes_out > byte_budget:
                        raise SseParseError("stream exceeded its duration or size budget")
                    for event in parser.feed(chunk):
                        events_in += 1
                        for block in adapter.process(event):
                            if first_byte_at is None:
                                first_byte_at = time.perf_counter()
                            bytes_out += len(block)
                            yield block
                for event in parser.flush():
                    events_in += 1
                    for block in adapter.process(event):
                        bytes_out += len(block)
                        yield block
                for block in adapter.finish():
                    bytes_out += len(block)
                    yield block
            except (httpx.HTTPError, SseParseError) as exc:
                outcome = "upstream_error"
                # Only the exception type is logged: h11 and httpx put raw upstream
                # bytes into their messages.
                logger.error(
                    "upstream stream failed",
                    extra={
                        "event": "stream.upstream_error",
                        "request_id": request_id,
                        "cause": type(exc).__name__,
                        "limit_hit": isinstance(exc, SseParseError),
                    },
                )
                # The status line is long gone; tell the client in-band and stop.
                yield serialize_event(
                    json.dumps(
                        error_payload(
                            "upstream_stream_error",
                            "The model provider stream ended unexpectedly",
                            request_id,
                        )
                    ),
                    event="error",
                )
            except Exception:
                # A defect in the gateway itself must never degrade into relaying
                # unredacted bytes: end the stream with an in-band error instead.
                outcome = "gateway_error"
                logger.exception(
                    "stream processing failed",
                    extra={"event": "stream.gateway_error", "request_id": request_id},
                )
                yield serialize_event(
                    json.dumps(
                        error_payload(
                            "gateway_stream_error",
                            "The gateway could not process the model provider stream",
                            request_id,
                        )
                    ),
                    event="error",
                )
            finally:
                await upstream.aclose()
                now = time.perf_counter()
                logger.info(
                    "stream finished",
                    extra={
                        "event": "stream.finished",
                        "request_id": request_id,
                        "dialect": dialect.name,
                        "outcome": outcome,
                        "upstream_ttfb_ms": upstream_ttfb_ms,
                        "ttft_ms": round((first_byte_at - started) * 1000, 2)
                        if first_byte_at
                        else None,
                        "duration_ms": round((now - started) * 1000, 2),
                        "events_in": events_in,
                        "bytes_out": bytes_out,
                        "redactions": dict(adapter.redaction_counts),
                        "stripped": dict(adapter.stripped),
                    },
                )

        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            HEADER_REQUEST_ID: request_id,
        }
        return StreamingResponse(body(), status_code=200, headers=headers)

    async def _buffered(
        self,
        upstream: httpx.Response,
        adapter: StreamAdapter,
        request_id: str,
        started: float,
        dialect: Dialect,
        success: bool,
    ) -> Response:
        limit = self._settings.max_response_bytes
        chunks: list[bytes] = []
        total = 0
        try:
            async for chunk in upstream.aiter_bytes():
                total += len(chunk)
                if total > limit:
                    raise GatewayError(
                        502,
                        "upstream_response_too_large",
                        f"The model provider response exceeds {limit} bytes",
                    )
                chunks.append(chunk)
        except httpx.HTTPError as exc:
            logger.error(
                "upstream read failed",
                extra={
                    "event": "upstream.read_error",
                    "request_id": request_id,
                    "cause": type(exc).__name__,
                },
            )
            return self._json(
                502,
                error_payload(
                    "upstream_unavailable", "The model provider is unavailable", request_id
                ),
                request_id,
            )
        except GatewayError as exc:
            return self._json(
                exc.status, error_payload(exc.error_type, exc.message, request_id), request_id
            )
        finally:
            await upstream.aclose()

        raw = b"".join(chunks)
        content_type = upstream.headers.get("content-type", "")
        try:
            if content_type.startswith("text/event-stream"):
                payload = _first_json_event(raw, self._settings.max_event_bytes)
            else:
                payload = load_json_document(raw) if raw else None
        except (ValueError, RecursionError, SseParseError):
            payload = None

        if not success:
            # Provider error documents are relayed when they are JSON (redacted, since
            # they may echo request text); anything else is replaced by a generic error.
            body: Any = (
                adapter.redact_envelope(payload)
                if isinstance(payload, dict)
                else error_payload(
                    "upstream_error",
                    f"The model provider returned HTTP {upstream.status_code}",
                    request_id,
                )
            )
            logger.warning(
                "upstream error relayed",
                extra={
                    "event": "upstream.error",
                    "request_id": request_id,
                    "status": upstream.status_code,
                },
            )
            relayed = {
                name: value
                for name, value in upstream.headers.items()
                if name.lower().startswith(_RELAYED_ERROR_HEADER_PREFIXES)
            }
            return self._json(upstream.status_code, body, request_id, relayed)

        if not isinstance(payload, dict | list):
            # Fail closed: a response the gateway cannot parse cannot be guaranteed clean.
            return self._json(
                502,
                error_payload(
                    "upstream_unparseable",
                    "The model provider returned an unreadable response",
                    request_id,
                ),
                request_id,
            )

        # A JSON array is the non-SSE streaming shape of Gemini: an array of chunks.
        redacted = adapter.redact_response(payload)
        logger.info(
            "response redacted",
            extra={
                "event": "response.finished",
                "request_id": request_id,
                "dialect": dialect.name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "redactions": dict(adapter.redaction_counts),
                "stripped": dict(adapter.stripped),
            },
        )
        return self._json(upstream.status_code, redacted, request_id)

    # -- endpoints -----------------------------------------------------------

    async def chat_completions(self, request: Request) -> Response:
        return await self._handle(request, OPENAI)

    async def messages(self, request: Request) -> Response:
        return await self._handle(request, ANTHROPIC)

    async def gemini(self, request: Request) -> Response:
        version = request.path_params.get("version", "")
        model_action = str(request.path_params.get("model_action", ""))
        model, _, action = model_action.rpartition(":")
        if version not in _GEMINI_VERSIONS or not model or action not in _GEMINI_ACTIONS:
            request_id = resolve_request_id(request.headers.get(HEADER_REQUEST_ID))
            return self._json(
                404,
                error_payload("not_found", "Unsupported Gemini endpoint", request_id),
                request_id,
            )
        return await self._handle(request, GEMINI)

    async def health(self, request: Request) -> Response:
        """Liveness only: no configuration details for anonymous callers."""
        return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

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
            "unhandled error", extra={"event": "gateway.error", "request_id": request_id}
        )
        return self._json(
            500, error_payload("internal_error", "Internal gateway error", request_id), request_id
        )


def _first_json_event(raw: bytes, max_event_bytes: int) -> Any:
    """The JSON object of the first ``data:`` event in an SSE body (provider error streams)."""
    parser = SseParser(max_event_bytes=max_event_bytes)
    for event in [*parser.feed(raw), *parser.flush()]:
        if event.has_data:
            return load_json_document(event.data)
    return None


def create_app(
    settings: GuardrailSettings, *, http_client: httpx.AsyncClient | None = None
) -> Starlette:
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.read_timeout_seconds,
            write=10.0,
            pool=5.0,
        ),
        follow_redirects=False,
    )
    service = GuardrailService(settings, client)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        logger.info(
            "guardrail gateway ready",
            extra={
                "event": "gateway.start",
                "upstream": redact_url(settings.upstream_base_url),
                "patterns": list(settings.pattern_names),
            },
        )
        try:
            yield
        finally:
            if owns_client:
                await client.aclose()

    app = Starlette(
        routes=[
            Route(OPENAI.fixed_path or "", service.chat_completions, methods=["POST"]),
            Route(ANTHROPIC.fixed_path or "", service.messages, methods=["POST"]),
            Route("/{version}/models/{model_action}", service.gemini, methods=["POST"]),
            Route("/healthz", service.health, methods=["GET"]),
        ],
        lifespan=lifespan,
        exception_handlers={HTTPException: service.http_error, Exception: service.unhandled_error},
    )
    # No unauthenticated 307 to "/path/" built from the client's Host header.
    app.router.redirect_slashes = False
    return app


def app_from_env() -> Starlette:
    """Application factory for ``uvicorn --factory llm_guardrail_gateway.app:app_from_env``."""
    settings = GuardrailSettings.from_env()
    configure_logging(settings.log_level)
    return create_app(settings)
