"""Downstream forwarding: header hygiene, identity propagation and relaying."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx
from starlette.datastructures import Headers

from mcp_gateway.auth import Principal

logger = logging.getLogger(__name__)

HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
IDENTITY_HEADER_PREFIX = "x-gateway-"
HEADER_SUBJECT = "X-Gateway-Subject"
HEADER_ROLE = "X-Gateway-Role"
HEADER_REQUEST_ID = "X-Request-Id"
HEADER_FORWARDED_FOR = "X-Forwarded-For"

# Request headers that never cross the gateway: hop-by-hop, framing, the
# caller's own credentials and anything a client could use to impersonate the
# gateway's identity assertions.
_DROPPED_REQUEST_HEADERS = HOP_BY_HOP_HEADERS | {
    "host",
    "content-length",
    "authorization",
    "cookie",
    HEADER_REQUEST_ID.lower(),
    HEADER_FORWARDED_FOR.lower(),
}
# Response headers the gateway's own server supplies; a downstream copy would
# either duplicate them or advertise the downstream's software to clients.
_DROPPED_RESPONSE_HEADERS = HOP_BY_HOP_HEADERS | {"server", "date"}


class DownstreamError(Exception):
    """The downstream could not be reached or did not answer in time."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


def _connection_tokens(headers: Headers | httpx.Headers) -> set[str]:
    raw = headers.get("connection", "")
    return {token.strip().lower() for token in raw.split(",") if token.strip()}


def build_forward_headers(
    incoming: Headers,
    *,
    principal: Principal,
    request_id: str,
    client_host: str | None,
    service_token: str | None,
) -> list[tuple[bytes, bytes]]:
    """Headers for the downstream request, as raw bytes.

    The caller's ``Authorization`` is removed and replaced by the gateway's
    own service credential (if configured); the authenticated identity travels
    in ``X-Gateway-*`` headers that clients cannot spoof because any incoming
    header with that prefix is dropped first. Client headers are relayed as
    the bytes they arrived in: decoding and re-encoding them would turn the
    odd non-ASCII value into a failure after the request was accepted.
    """
    dropped = _DROPPED_REQUEST_HEADERS | _connection_tokens(incoming)
    forwarded: list[tuple[bytes, bytes]] = []
    for raw_name, value in incoming.raw:
        name = raw_name.decode("latin-1").lower()
        if name not in dropped and not name.startswith(IDENTITY_HEADER_PREFIX):
            forwarded.append((raw_name.lower(), value))
    forwarded.append((HEADER_SUBJECT.lower().encode("ascii"), principal.subject.encode("ascii")))
    forwarded.append((HEADER_ROLE.lower().encode("ascii"), principal.role.encode("ascii")))
    forwarded.append((HEADER_REQUEST_ID.lower().encode("ascii"), request_id.encode("ascii")))

    chain = [p.strip() for p in incoming.get(HEADER_FORWARDED_FOR, "").split(",") if p.strip()]
    if client_host:
        chain.append(client_host)
    if chain:
        forwarded.append(
            (HEADER_FORWARDED_FOR.lower().encode("ascii"), ", ".join(chain).encode("latin-1"))
        )

    if service_token:
        forwarded.append((b"authorization", f"Bearer {service_token}".encode("ascii")))
    return forwarded


def filter_response_headers(upstream: httpx.Response) -> list[tuple[bytes, bytes]]:
    """Relay the downstream response headers as raw bytes.

    Hop-by-hop headers are dropped, as are ``Server`` and ``Date``, which the
    gateway's own server supplies. The body is relayed as raw bytes, so
    ``Content-Length`` and ``Content-Encoding`` stay accurate and are kept.
    Values are relayed as received: re-encoding them would fail on the odd
    non-Latin-1 header.
    """
    dropped = _DROPPED_RESPONSE_HEADERS | _connection_tokens(upstream.headers)
    return [
        (name.lower(), value)
        for name, value in upstream.headers.raw
        if name.decode("latin-1").lower() not in dropped
    ]


class DownstreamClient:
    """Thin wrapper over ``httpx.AsyncClient`` that maps transport failures."""

    def __init__(self, http_client: httpx.AsyncClient, downstream_url: str) -> None:
        self._client = http_client
        self._url = downstream_url

    @property
    def url(self) -> str:
        return self._url

    async def send(
        self,
        method: str,
        headers: Sequence[tuple[bytes, bytes]],
        content: bytes | None,
        query: str = "",
        *,
        timeouts: httpx.Timeout | None = None,
    ) -> httpx.Response:
        """Send one request; ``timeouts`` overrides the client's defaults when given."""
        url = self._url
        if query:
            url = f"{url}{'&' if '?' in url else '?'}{query}"
        request = self._client.build_request(
            method,
            url,
            headers=list(headers),
            content=content,
            timeout=httpx.USE_CLIENT_DEFAULT if timeouts is None else timeouts,
        )
        try:
            return await self._client.send(request, stream=True)
        except httpx.PoolTimeout as exc:
            # Not a downstream fault: the gateway's own connection pool is full.
            raise DownstreamError("overloaded", f"{type(exc).__name__}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise DownstreamError("timeout", f"{type(exc).__name__}: {exc}") from exc
        except httpx.TransportError as exc:
            raise DownstreamError("unavailable", f"{type(exc).__name__}: {exc}") from exc
