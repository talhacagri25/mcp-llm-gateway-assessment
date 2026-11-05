"""Structured JSON logging to stderr with bounded, non-forgeable records."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_STANDARD_RECORD_ATTRIBUTES = frozenset(
    logging.LogRecord("", logging.INFO, "", 0, "", (), None).__dict__
) | {"message", "asctime"}
MAX_FIELD_CHARS = 512
"""Longest string value written to a log line; longer values are cut and marked."""
MAX_ITEMS = 64
"""Longest list or mapping written to a log line."""


def bound(value: Any, depth: int = 0) -> Any:
    """Cap string lengths and container sizes so one request cannot flood the log.

    Caller controlled values (tool names, subjects, model names) end up in
    structured fields; JSON encoding already keeps them on one line, this
    keeps the line short and the volume per request bounded.
    """
    if isinstance(value, str):
        if len(value) <= MAX_FIELD_CHARS:
            return value
        return f"{value[:MAX_FIELD_CHARS]}...[{len(value) - MAX_FIELD_CHARS} more chars]"
    if depth >= 6:
        return "[nested]"
    if isinstance(value, Mapping):
        items = list(value.items())
        out = {str(k)[:64]: bound(v, depth + 1) for k, v in items[:MAX_ITEMS]}
        if len(items) > MAX_ITEMS:
            out["..."] = f"[{len(items) - MAX_ITEMS} more items]"
        return out
    if isinstance(value, list | tuple | set | frozenset):
        items = list(value)
        out_list = [bound(v, depth + 1) for v in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            out_list.append(f"[{len(items) - MAX_ITEMS} more items]")
        return out_list
    if isinstance(value, int | float | bool) or value is None:
        return value
    return bound(str(value), depth)


def redact_url(url: str) -> str:
    """A URL safe to log: user information removed, query string dropped."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "[unparseable url]"
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    if "@" in parts.netloc:
        host = f"***@{host}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": bound(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRIBUTES and not key.startswith("_"):
                payload[key] = bound(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # ensure_ascii keeps every record a single line of printable ASCII, so a
        # terminal or a log shipper can never be confused by what a caller sent.
        return json.dumps(payload, default=str, ensure_ascii=True)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # httpx logs every request line at INFO, URL and query string included (a
    # ``?key=`` credential among them); the services write their own redacted
    # records instead, so the HTTP client libraries only get to report problems.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
