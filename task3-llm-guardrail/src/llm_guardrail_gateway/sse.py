"""Incremental Server-Sent Events parsing and serialisation.

The parser is fed raw bytes as they arrive from the upstream socket and
yields complete events as soon as their terminating blank line is seen. Each
event keeps its exact original bytes for diagnostics, but adapters never
relay those bytes: every event is rebuilt from its parsed fields with the
field values sanitised, so an upstream cannot hide a second ``data:`` line
behind a character some client libraries treat as a line break (U+2028,
U+2029, U+0085, the ASCII separators 0x1c to 0x1e) and cannot smuggle text
through ``id:``, ``event:`` or comment lines unredacted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# Two consecutive line endings, each CRLF, LF or a CR that does not start a CRLF.
_TERMINATOR: Final = re.compile(rb"(?:\r\n|\n|\r(?!\n)){2}")
_LINE_BREAK: Final = re.compile(r"\r\n|\n|\r")
_BOM: Final = b"\xef\xbb\xbf"
_FIELD_BREAKS: Final = re.compile("[\r\n\u2028\u2029\x85\x1c\x1d\x1e]")
"""Characters that end a line for some SSE client libraries; never emitted inside a field."""
DEFAULT_MAX_EVENT_BYTES: Final = 256 * 1024


class SseParseError(Exception):
    """The stream violated a bound the gateway enforces."""


@dataclass(slots=True)
class SseEvent:
    raw: bytes
    """The event block exactly as received, including its terminator (diagnostics only)."""
    data: str
    """All ``data`` lines joined with newlines (empty when the block had none)."""
    event: str | None = None
    id: str | None = None
    retry: str | None = None
    has_data: bool = False
    comments: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return f"SseEvent(event={self.event!r}, data={self.data[:60]!r})"


def _parse_block(block: bytes) -> SseEvent:
    text = block.decode("utf-8", errors="replace")
    data_lines: list[str] = []
    comments: list[str] = []
    event = event_id = retry = None
    has_data = False
    for line in _LINE_BREAK.split(text):
        if not line:
            continue
        if line.startswith(":"):
            comments.append(line[2:] if line.startswith(": ") else line[1:])
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
            has_data = True
        elif field == "event":
            event = value
        elif field == "id":
            event_id = value
        elif field == "retry":
            retry = value
    return SseEvent(
        raw=block,
        data="\n".join(data_lines),
        event=event,
        id=event_id,
        retry=retry,
        has_data=has_data,
        comments=tuple(comments),
    )


class SseParser:
    def __init__(self, max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES) -> None:
        self._buffer = bytearray()
        self._scan_from = 0
        self._max_event_bytes = max_event_bytes
        self._bom_checked = False

    def feed(self, chunk: bytes) -> list[SseEvent]:
        """Append ``chunk`` and return every event completed by it."""
        self._buffer += chunk
        if not self._bom_checked and len(self._buffer) >= len(_BOM):
            # The SSE specification requires one leading byte order mark to be
            # ignored; left in place it would turn "data" into an unknown field
            # and let the first event bypass redaction.
            self._bom_checked = True
            if self._buffer.startswith(_BOM):
                del self._buffer[: len(_BOM)]
        events: list[SseEvent] = []
        while True:
            match = _TERMINATOR.search(self._buffer, self._scan_from)
            if match is None:
                if len(self._buffer) > self._max_event_bytes:
                    raise SseParseError(f"SSE event exceeds the {self._max_event_bytes} byte limit")
                # A terminator may be split across chunks: rescan its last bytes next time.
                self._scan_from = max(0, len(self._buffer) - 3)
                return events
            block = bytes(self._buffer[: match.end()])
            del self._buffer[: match.end()]
            self._scan_from = 0
            events.append(_parse_block(block))

    def flush(self) -> list[SseEvent]:
        """Return a trailing event that was never terminated (lenient end of stream)."""
        if not self._buffer.strip():
            self._buffer.clear()
            return []
        block = bytes(self._buffer)
        self._buffer.clear()
        self._scan_from = 0
        return [_parse_block(block)]


def sanitize_field(value: str) -> str:
    """A field value that cannot break out of its line, whatever the client's line rules."""
    return _FIELD_BREAKS.sub(" ", value)


def serialize_event(
    data: str | None,
    *,
    event: str | None = None,
    event_id: str | None = None,
    retry: str | None = None,
    comments: tuple[str, ...] = (),
) -> bytes:
    """Serialise one event; ``data`` may contain newlines (one ``data:`` line each)."""
    lines: list[str] = [f": {sanitize_field(comment)}" for comment in comments]
    if event is not None:
        lines.append(f"event: {sanitize_field(event)}")
    if event_id is not None:
        lines.append(f"id: {sanitize_field(event_id)}")
    if retry is not None:
        lines.append(f"retry: {sanitize_field(retry)}")
    if data is not None:
        lines.extend(f"data: {sanitize_field(part)}" for part in data.split("\n"))
    if not lines:
        return b""
    # A lone surrogate (which JSON escapes can express) must not kill the stream.
    return ("\n".join(lines) + "\n\n").encode("utf-8", errors="replace")
