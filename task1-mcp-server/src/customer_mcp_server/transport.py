"""Transport level protocol compliance helpers.

The SDK's stdio transport turns a line that cannot be decoded into a JSON-RPC
message into an ``Exception`` item on the read stream and the serving loop
silently drops it. JSON-RPC 2.0 requires a response in that situation:
``-32700 Parse error`` for invalid JSON and ``-32600 Invalid Request`` for
JSON that is not a request object, both with ``"id": null``. This module wraps
the read stream so those frames are answered on the wire and recorded on
stderr while every well formed message passes through untouched.

Two refinements on top of the SDK's decoding:

* a request whose ``id`` is not a string or integer (``1.5``, ``true``,
  ``null``, an array) validates as a *notification* in the SDK because the
  ``id`` member is simply ignored; left alone, the client would wait forever.
  MCP reserves the ``notifications/`` prefix for notifications, so a
  notification carrying any other method is answered with ``-32600``;
* when an invalid request does carry a usable id, the error echoes it instead
  of ``null`` (JSON-RPC reserves ``null`` for ids that cannot be determined).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from types import TracebackType
from typing import Any, BinaryIO, Self

import anyio
import anyio.to_thread
import mcp_types as types
from mcp.shared._stream_protocols import ReadStream, WriteStream
from mcp.shared.message import SessionMessage
from pydantic import ValidationError

logger = logging.getLogger(__name__)

_JSON_SYNTAX_ERROR_TYPES = frozenset({"json_invalid", "json_type"})
_NOTIFICATION_PREFIX = "notifications/"


MAX_ECHOED_CHARS = 256
"""Longest caller supplied string echoed back in an error produced on the wire."""
DEFAULT_MAX_FRAME_BYTES = 4 * 1024 * 1024
RESPONSE_GRACE_SECONDS = 5.0


def recoverable_id(exc: ValidationError) -> str | int | None:
    """The request id of the offending frame when it is a valid string or integer.

    Only the frame itself is consulted (an object carrying ``jsonrpc``); an
    ``id`` nested inside a malformed id value must not be mistaken for it.
    """
    for err in exc.errors(include_url=False, include_input=True):
        document = err.get("input")
        if isinstance(document, dict) and "jsonrpc" in document and "id" in document:
            candidate = document["id"]
            if isinstance(candidate, str) or (
                isinstance(candidate, int) and not isinstance(candidate, bool)
            ):
                return candidate
            return None
    return None


def bound_value(value: Any, limit: int = MAX_ECHOED_CHARS, depth: int = 0) -> Any:
    """Cap every string inside ``value`` so an error can never echo a megabyte of input."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    if depth >= 6:
        return "[nested]"
    if isinstance(value, dict):
        return {str(k)[:64]: bound_value(v, limit, depth + 1) for k, v in list(value.items())[:64]}
    if isinstance(value, list | tuple):
        return [bound_value(v, limit, depth + 1) for v in list(value)[:64]]
    return value


def classify_decode_failure(exc: Exception) -> tuple[int, str, str | int | None]:
    """Map a transport decode failure to the JSON-RPC error (code, message, id) it must produce."""
    if isinstance(exc, ValidationError):
        error_types = {err["type"] for err in exc.errors(include_url=False, include_input=False)}
        if error_types & _JSON_SYNTAX_ERROR_TYPES:
            return types.PARSE_ERROR, "Parse error", None
        return types.INVALID_REQUEST, "Invalid Request", recoverable_id(exc)
    return types.PARSE_ERROR, "Parse error", None


def is_request_without_valid_id(item: SessionMessage) -> bool:
    """A notification whose method is a request method: its id was dropped as malformed."""
    message = item.message
    return isinstance(message, types.JSONRPCNotification) and not message.method.startswith(
        _NOTIFICATION_PREFIX
    )


class ShieldedWriteStream:
    """Write stream decorator that lets an answer in flight survive end of input.

    When the client half-closes stdin right after its last request (a pipe
    style host), the SDK cancels the in-flight handler tasks. A handler that
    has already executed a refund and is parked in ``send`` would then lose
    its receipt: the money moved, the client never hears about it. Sends run
    in a shielded cancel scope bounded by a grace period instead. The
    decorator also bounds every string in outgoing error frames, so caller
    text that the SDK echoes (an unknown method name, for example) can never
    amplify a request into a megabyte response.
    """

    def __init__(
        self, inner: WriteStream[SessionMessage], grace_seconds: float = RESPONSE_GRACE_SECONDS
    ) -> None:
        self._inner = inner
        self._grace = grace_seconds

    async def send(self, item: SessionMessage) -> None:
        message = item.message
        if isinstance(message, types.JSONRPCError):
            error = message.error
            bounded = types.ErrorData(
                code=error.code,
                message=bound_value(error.message),
                data=bound_value(error.data),
            )
            item = SessionMessage(
                types.JSONRPCError(jsonrpc="2.0", id=message.id, error=bounded),
                metadata=item.metadata,
            )
        with anyio.CancelScope(shield=True), anyio.move_on_after(self._grace):
            await self._inner.send(item)

    async def send_protocol_error(self, code: int, text: str) -> None:
        frame = types.JSONRPCError(
            jsonrpc="2.0", id=None, error=types.ErrorData(code=code, message=text)
        )
        try:
            await self.send(SessionMessage(frame))
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            logger.debug("write stream closed while reporting a protocol error")

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> Self:
        await self._inner.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return await self._inner.__aexit__(exc_type, exc_val, exc_tb)


class BoundedLineReader:
    """Reads stdin line by line with a hard cap on the size of one frame.

    The SDK reads lines without a limit, so a host could make the server buffer
    and parse frames of arbitrary size. Lines above the cap are discarded
    whole, answered with ``-32600``, and never reach the JSON parser. Reads
    happen in worker threads that are abandoned on cancellation, so shutdown
    never waits for the client to send another line.
    """

    def __init__(self, raw: BinaryIO, max_line_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> None:
        self._raw = raw
        self._max = max_line_bytes
        self._report: Callable[[int, str], Awaitable[None]] | None = None

    def set_reporter(self, report: Callable[[int, str], Awaitable[None]]) -> None:
        self._report = report

    async def _readline(self, limit: int) -> bytes:
        result: bytes = await anyio.to_thread.run_sync(
            self._raw.readline, limit, abandon_on_cancel=True
        )
        return result

    async def __aiter__(self) -> AsyncIterator[str]:
        while True:
            line = await self._readline(self._max + 1)
            if not line:
                return
            if len(line) > self._max:
                dropped = len(line)
                while line and not line.endswith(b"\n"):
                    line = await self._readline(1 << 20)
                    dropped += len(line)
                logger.warning(
                    "oversized frame discarded",
                    extra={"event": "wire.frame_too_large", "bytes": dropped, "limit": self._max},
                )
                if self._report is not None:
                    await self._report(
                        types.INVALID_REQUEST,
                        f"Invalid Request: frame exceeds {self._max} bytes",
                    )
                continue
            yield line.decode("utf-8", errors="replace")


class ProtocolErrorReportingReadStream:
    """Read stream decorator that answers undecodable frames with JSON-RPC errors."""

    def __init__(
        self,
        inner: ReadStream[SessionMessage | Exception],
        write_stream: WriteStream[SessionMessage] | ShieldedWriteStream,
    ) -> None:
        self._inner = inner
        self._write_stream = write_stream

    @property
    def last_context(self) -> Any:
        # The serving loop reads the sender context through this attribute.
        return getattr(self._inner, "last_context", None)

    async def _report(self, exc: Exception) -> None:
        code, message, request_id = classify_decode_failure(exc)
        logger.warning(
            "rejected undecodable frame",
            extra={"event": "wire.undecodable", "code": code, "cause": type(exc).__name__},
        )
        await self._send_error(request_id, code, message)

    async def _report_missing_id(self, item: SessionMessage) -> None:
        method = getattr(item.message, "method", "?")
        logger.warning(
            "request without a valid id",
            extra={"event": "wire.invalid_request_id", "method": bound_value(method, 80)},
        )
        await self._send_error(
            None,
            types.INVALID_REQUEST,
            "Invalid Request: a request needs a string or integer id",
        )

    async def _send_error(self, request_id: str | int | None, code: int, message: str) -> None:
        frame = types.JSONRPCError(
            jsonrpc="2.0", id=request_id, error=types.ErrorData(code=code, message=message)
        )
        try:
            await self._write_stream.send(SessionMessage(frame))
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            logger.debug("write stream closed while reporting a protocol error")

    async def _filter(self, item: SessionMessage | Exception) -> SessionMessage | None:
        """Answer protocol violations on the wire; return the item when it should flow on."""
        if isinstance(item, Exception):
            await self._report(item)
            return None
        if is_request_without_valid_id(item):
            await self._report_missing_id(item)
            return None
        return item

    async def receive(self) -> SessionMessage | Exception:
        while True:
            item = await self._filter(await self._inner.receive())
            if item is not None:
                return item

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> SessionMessage | Exception:
        while True:
            item = await self._filter(await self._inner.__anext__())
            if item is not None:
                return item

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> Self:
        await self._inner.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return await self._inner.__aexit__(exc_type, exc_val, exc_tb)
