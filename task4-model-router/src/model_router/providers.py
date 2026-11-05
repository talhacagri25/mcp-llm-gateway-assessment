"""HTTP client for one model provider with a hard per-attempt deadline."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from model_router.config import ProviderSettings
from model_router.dialects import ProviderDialect, dialect_for

DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
"""A provider body larger than this is not buffered; the attempt counts as unparseable."""


class ProviderTimeoutError(Exception):
    pass


class ProviderUnavailableError(Exception):
    pass


class ProviderSaturatedError(Exception):
    """The provider's configured concurrency is fully in use right now."""


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    status: int
    body: Any
    """For 2xx: the response normalised to the Chat Completions shape (``None`` when not
    parseable). For errors: the provider's JSON error document, or ``None``."""
    raw_length: int
    latency_ms: float
    retry_after: str | None = None
    model: str | None = None
    """The model identifier that was sent to the provider."""


class ProviderClient:
    def __init__(
        self,
        settings: ProviderSettings,
        http_client: httpx.AsyncClient,
        *,
        forward_headers: Sequence[str] = (),
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.settings = settings
        self._client = http_client
        self.dialect: ProviderDialect = dialect_for(
            settings.dialect, max_tokens_field=settings.max_tokens_field
        )
        self._forward_headers = frozenset(h.lower() for h in forward_headers)
        self._max_response_bytes = max_response_bytes
        self._inflight = 0

    @property
    def name(self) -> str:
        return self.settings.name

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def failover_statuses(self) -> frozenset[int]:
        return self.dialect.extra_failover_statuses

    def resolve_model(self, payload: dict[str, Any], tenant_model: str | None) -> str | None:
        """Tenant pin, then provider pin, then whatever the client asked for."""
        if tenant_model:
            return tenant_model
        if self.settings.model:
            return self.settings.model
        model = payload.get("model")
        return model if isinstance(model, str) and model else None

    async def complete(
        self,
        payload: dict[str, Any],
        request_id: str,
        *,
        model: str | None,
        default_max_tokens: int,
        client_headers: Mapping[str, str] | None = None,
    ) -> ProviderResponse:
        """One attempt, bounded by the provider's ``timeout_ms`` from send to full body.

        Raises ``UnsupportedRequestError`` (from the dialect) before contacting the
        provider, ``ProviderSaturatedError`` when the configured concurrency is
        exhausted, and the timeout/unavailable errors for transport failures.
        """
        limit = self.settings.max_concurrency
        if limit and self._inflight >= limit:
            raise ProviderSaturatedError(f"{self.name}: {limit} requests already in flight")
        prepared = self.dialect.prepare(
            payload,
            model=model,
            api_key=self.settings.api_key,
            request_id=request_id,
            default_max_tokens=default_max_tokens,
        )
        headers = dict(prepared.headers)
        for name, value in (client_headers or {}).items():
            # Only visible ASCII survives: anything else cannot be a valid tracing header
            # and would fail on the way out, after the request was admitted and reserved.
            if name.lower() in self._forward_headers and _header_safe(value):
                headers.setdefault(name.lower(), value)
        started = time.perf_counter()
        request = self._client.build_request(
            "POST",
            f"{self.settings.base_url}{prepared.path}",
            headers=headers,
            content=json.dumps(prepared.body).encode("utf-8"),
        )
        self._inflight += 1
        try:
            # asyncio.timeout bounds the whole exchange (connect, send, headers, body)
            # even when the transport has no network timeouts of its own, and cancels
            # the in-flight request so a slow provider never blocks the fallback.
            async with asyncio.timeout(self.settings.timeout_ms / 1000):
                response = await self._client.send(request, stream=True)
                try:
                    content, truncated = await _read_capped(response, self._max_response_bytes)
                finally:
                    await response.aclose()
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"{self.name}: no response within {self.settings.timeout_ms} ms"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"{self.name}: {type(exc).__name__}") from exc
        except httpx.HTTPError as exc:
            # Transport failures, but also protocol and decoding errors (a body that
            # does not match its Content-Encoding): the provider gave no usable answer.
            raise ProviderUnavailableError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        finally:
            self._inflight -= 1

        body = None if truncated else _json_body(response.headers.get("content-type", ""), content)
        if 200 <= response.status_code < 300 and body is not None:
            body = self.dialect.normalize(body, requested_model=model)
        return ProviderResponse(
            status=response.status_code,
            body=body,
            raw_length=len(content),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            retry_after=response.headers.get("retry-after"),
            model=model,
        )


def _header_safe(value: str) -> bool:
    """Visible ASCII (plus space and tab) only: the field-value grammar of HTTP."""
    return all(0x20 <= ord(ch) < 0x7F or ch == "\t" for ch in value)


async def _read_capped(response: httpx.Response, limit: int) -> tuple[bytes, bool]:
    """Read at most ``limit`` decoded bytes of the body; ``True`` when it was larger.

    Decoded, not wire, bytes: the cap exists to bound what is buffered and
    parsed, which a compressed body could otherwise inflate past any limit.
    A body that does not match its ``Content-Encoding`` raises ``httpx.DecodingError``.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            return b"", True
        chunks.append(chunk)
    return b"".join(chunks), False


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _json_body(content_type: str, content: bytes) -> Any:
    """The parsed JSON body, or ``None`` when it is not JSON the gateway can relay.

    ``NaN``/``Infinity`` tokens, pathological nesting, oversized bodies and
    non-JSON media types all yield ``None``, which the router treats as an
    unparseable answer.
    """
    media_type = content_type.split(";")[0].strip().lower()
    if not (media_type.endswith("/json") or media_type.endswith("+json")):
        return None
    try:
        return json.loads(content, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None


def parse_retry_after(value: str | None, *, cap_seconds: float) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta seconds form), capped."""
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None  # the HTTP-date form is not worth parsing for a cooldown hint
    if seconds <= 0:
        return None
    return min(seconds, cap_seconds)
