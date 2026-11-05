"""JSON-RPC 2.0 wire format handling for the gateway.

The gateway needs to understand exactly enough of the wire format to locate
the method and params of every message (for policy decisions), to preserve
request ids when it answers on behalf of the downstream, and to reject
malformed input with the standard error codes. Messages that are forwarded
are relayed byte for byte; nothing here rewrites them.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Final

JSONRPC_VERSION: Final = "2.0"
MAX_NAME_LENGTH: Final = 256
"""Longest method or tool name the gateway examines. Names are compared after Unicode
normalisation, whose cost grows with length; a bound keeps that cost trivial and keeps
every echoed name short."""

# Standard JSON-RPC 2.0 codes
PARSE_ERROR: Final = -32700
INVALID_REQUEST: Final = -32600
METHOD_NOT_FOUND: Final = -32601
INVALID_PARAMS: Final = -32602
INTERNAL_ERROR: Final = -32603

# Gateway specific codes (implementation defined range -32000..-32099)
UNAUTHORIZED_TOOL_CALL: Final = -32001
UNAUTHENTICATED: Final = -32002
DOWNSTREAM_UNAVAILABLE: Final = -32003
DOWNSTREAM_TIMEOUT: Final = -32004
GATEWAY_OVERLOADED: Final = -32005
"""The gateway itself refused the request for capacity reasons (per-caller cap, pool exhausted)."""

RequestId = str | int | float | None

_MISSING: Final = object()


class JsonRpcProtocolError(Exception):
    """A payload that violates the JSON-RPC 2.0 structure."""

    def __init__(
        self,
        code: int,
        message: str,
        *,
        request_id: RequestId = None,
        data: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id
        self.data = data


@dataclass(frozen=True, slots=True)
class JsonRpcMessage:
    raw: dict[str, Any]
    method: str | None
    """``None`` when the message is a response travelling client to server."""
    params: Any
    request_id: RequestId
    has_id: bool
    """``False`` for notifications (the ``id`` member is absent, not null)."""

    @property
    def is_request(self) -> bool:
        return self.method is not None and self.has_id

    @property
    def is_notification(self) -> bool:
        return self.method is not None and not self.has_id

    @property
    def is_response(self) -> bool:
        return self.method is None


@dataclass(frozen=True, slots=True)
class JsonRpcPayload:
    messages: tuple[JsonRpcMessage, ...]
    is_batch: bool

    @property
    def requests(self) -> tuple[JsonRpcMessage, ...]:
        return tuple(m for m in self.messages if m.is_request)


class DuplicateMemberError(ValueError):
    """A JSON object repeated a member name."""


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # Python keeps the last duplicate, other parsers keep the first. The gateway
    # forwards the original bytes, so the only safe policy decision is to refuse
    # documents whose meaning depends on the parser.
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise DuplicateMemberError(key)
        document[key] = value
    return document


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def clean_identifier(text: str) -> bool:
    """``False`` when a name carries control characters or lone surrogates.

    Such names cannot be normalised and compared meaningfully, and some
    downstream parsers reject or rewrite them; the gateway refuses them
    instead of guessing what the downstream would see.
    """
    return not any(unicodedata.category(ch) in ("Cc", "Cs") for ch in text)


def parse_payload(body: bytes, *, max_batch_size: int | None = None) -> JsonRpcPayload:
    """Parse a request body into messages, raising ``JsonRpcProtocolError`` when malformed.

    A batch with any malformed element is rejected as a whole: the gateway
    forwards batches verbatim, so it cannot answer one element and forward the
    rest without re-assembling responses. Duplicate object members, ``NaN`` and
    ``Infinity`` tokens and absurdly long numbers are rejected rather than
    interpreted, because the downstream parser may read them differently.
    """
    try:
        # Decoded explicitly as UTF-8 (the JSON-RPC and MCP wire encoding): json.loads
        # would otherwise sniff UTF-16 and UTF-32 bodies that the downstream, which
        # receives the original bytes, may read as something else entirely.
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JsonRpcProtocolError(PARSE_ERROR, "Parse error: body must be UTF-8") from exc
    try:
        decoded = json.loads(
            text, object_pairs_hook=_pairs_without_duplicates, parse_constant=_reject_constant
        )
    except DuplicateMemberError as exc:
        raise JsonRpcProtocolError(
            INVALID_REQUEST, "Invalid Request: duplicate member in JSON object"
        ) from exc
    except RecursionError as exc:
        # Thousands of nested brackets exhaust the decoder's recursion budget;
        # that is a malformed document, not a gateway fault.
        raise JsonRpcProtocolError(PARSE_ERROR, "Parse error: document nested too deeply") from exc
    except ValueError as exc:
        # Also covers the interpreter's integer digit limit and a leading BOM.
        raise JsonRpcProtocolError(PARSE_ERROR, "Parse error") from exc

    if isinstance(decoded, list):
        if not decoded:
            raise JsonRpcProtocolError(INVALID_REQUEST, "Invalid Request: empty batch")
        if max_batch_size is not None and len(decoded) > max_batch_size:
            raise JsonRpcProtocolError(
                INVALID_REQUEST, f"Invalid Request: batch exceeds {max_batch_size} messages"
            )
        return JsonRpcPayload(tuple(_parse_message(item) for item in decoded), is_batch=True)
    return JsonRpcPayload((_parse_message(decoded),), is_batch=False)


def _parse_message(item: Any) -> JsonRpcMessage:
    if not isinstance(item, dict):
        raise JsonRpcProtocolError(INVALID_REQUEST, "Invalid Request: message must be an object")

    raw_id = item.get("id", _MISSING)
    has_id = raw_id is not _MISSING
    if has_id and (isinstance(raw_id, bool) or not isinstance(raw_id, str | int | float | None)):
        raise JsonRpcProtocolError(
            INVALID_REQUEST, "Invalid Request: id must be a string, a number or null"
        )
    if isinstance(raw_id, float) and not math.isfinite(raw_id):
        # 1e400 parses as infinity, which no response could carry back.
        raise JsonRpcProtocolError(INVALID_REQUEST, "Invalid Request: id must be a finite number")
    request_id: RequestId = raw_id if has_id else None

    if item.get("jsonrpc") != JSONRPC_VERSION:
        raise JsonRpcProtocolError(
            INVALID_REQUEST, 'Invalid Request: jsonrpc must be "2.0"', request_id=request_id
        )

    if "method" in item:
        method = item["method"]
        if not isinstance(method, str) or not method:
            raise JsonRpcProtocolError(
                INVALID_REQUEST,
                "Invalid Request: method must be a non-empty string",
                request_id=request_id,
            )
        if len(method) > MAX_NAME_LENGTH:
            raise JsonRpcProtocolError(
                INVALID_REQUEST,
                f"Invalid Request: method name exceeds {MAX_NAME_LENGTH} characters",
                request_id=request_id,
            )
        if not clean_identifier(method):
            raise JsonRpcProtocolError(
                INVALID_REQUEST,
                "Invalid Request: method name contains control characters",
                request_id=request_id,
            )
        params = item.get("params")
        if params is not None and not isinstance(params, dict | list):
            raise JsonRpcProtocolError(
                INVALID_REQUEST,
                "Invalid Request: params must be an object or an array",
                request_id=request_id,
            )
        return JsonRpcMessage(
            raw=item, method=method, params=params, request_id=request_id, has_id=has_id
        )

    if has_id and ("result" in item or "error" in item):
        # A response from the client to a server initiated request; forwarded as-is.
        return JsonRpcMessage(
            raw=item, method=None, params=None, request_id=request_id, has_id=True
        )

    raise JsonRpcProtocolError(
        INVALID_REQUEST, "Invalid Request: missing method", request_id=request_id
    )


def error_response(
    request_id: RequestId, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}
