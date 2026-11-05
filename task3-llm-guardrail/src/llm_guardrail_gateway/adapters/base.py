"""Common machinery for provider adapters.

An adapter knows where the model's text lives inside a provider's SSE events
and JSON responses. It owns one ``StreamingRedactor`` per *channel* (for
example per choice index, or per tool call) because each channel is an
independent text stream whose chunk boundaries must be tracked separately.

Everything an adapter does not recognise is treated as suspect rather than
trusted (fail closed):

* string fields outside the known text channels are redacted in one shot per
  event, except a short list of opaque identifiers (ids, model names, tool
  names, cryptographic signatures, base64 payloads) that cannot carry PII and
  must not be altered;
* tool inputs and function arguments are redacted exhaustively, whatever
  their keys are called, because their shape is defined by the caller;
* ``data`` events that are not JSON are redacted as plain text;
* token-level ``logprobs`` are removed, because they would reproduce the
  redacted text one token at a time;
* every event is rebuilt from its parsed fields (never relayed byte for
  byte), with comment, ``id`` and ``event`` values redacted and sanitised;
* a field a handler expected but found in another type (a list where a
  string belongs, a string where an object belongs) is redacted exhaustively
  instead of passing through: a failed type check is never a pass.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Hashable, Iterable
from typing import Any, Final

from llm_guardrail_gateway.redaction.streaming import StreamingRedactor
from llm_guardrail_gateway.sse import SseEvent, SseParser, serialize_event

RedactorFactory = Callable[..., StreamingRedactor]
"""Builds a ``StreamingRedactor``; called with ``json_mode=True`` for JSON text channels."""

_COMPACT = {"ensure_ascii": False, "separators": (",", ":")}
_JSON_LINE_BREAK_ESCAPES: Final = str.maketrans(
    {
        "\u2028": "\\u2028",
        "\u2029": "\\u2029",
        "\x85": "\\u0085",
        "\x1c": "\\u001c",
        "\x1d": "\\u001d",
        "\x1e": "\\u001e",
    }
)
"""Characters some SSE clients treat as line breaks, escaped inside JSON strings (where
alone they can occur in ``json.dumps`` output) so a ``data:`` payload stays one line."""
MAX_CHANNELS: Final = 64
"""Distinct text channels per response before new ones share an overflow channel."""
_OVERFLOW_KEY: Final = ("overflow", None, None)

OPAQUE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "type",
        "object",
        "model",
        "role",
        "index",
        "created",
        "finish_reason",
        "stop_reason",
        "system_fingerprint",
        "service_tier",
        "signature",
        "data",
        "encrypted_content",
        "encrypted_index",
        "media_type",
        "mimeType",
        "mime_type",
        "finishReason",
        "modelVersion",
        "responseId",
        "thoughtSignature",
        "fileUri",
    }
)
"""Envelope fields that are identifiers or opaque material, never prose."""

CONTENT_KEYS: Final[frozenset[str]] = frozenset({"input", "arguments", "args", "response"})
"""Caller-shaped containers: every string inside them is redacted, whatever its key."""


def dump_json(payload: Any) -> str:
    return json.dumps(payload, **_COMPACT).translate(_JSON_LINE_BREAK_ESCAPES)  # type: ignore[arg-type]


def normalize_index(value: Any) -> int | str:
    """Channel index as the provider meant it: ``"0"`` and ``0`` are the same choice."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return str(value)[:64]


def parse_json(text: str) -> Any:
    """``json.loads`` that treats every failure (including pathological nesting) as ``None``."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _walk(value: Any, redact: Callable[[str], str], *, full: bool) -> tuple[Any, bool]:
    """Return a copy of ``value`` with its strings redacted and whether anything changed.

    Integers are checked as well: a card number emitted as a JSON number is
    still a card number. It comes back as the replacement string.
    """
    if isinstance(value, str):
        replaced = redact(value)
        return replaced, replaced != value
    if isinstance(value, int | float) and not isinstance(value, bool):
        integral = isinstance(value, float) and value.is_integer() and abs(value) < 2**53
        text = str(int(value)) if integral else repr(value)
        replaced = redact(text)
        return (replaced, True) if replaced != text else (value, False)
    if isinstance(value, list):
        changed = False
        items: list[Any] = []
        for item in value:
            new_item, item_changed = _walk(item, redact, full=full)
            items.append(new_item)
            changed = changed or item_changed
        return items, changed
    if isinstance(value, dict):
        changed = False
        mapping: dict[Any, Any] = {}
        for key, item in value.items():
            if not full and key in OPAQUE_KEYS:
                mapping[key] = item
                continue
            new_key = key
            if full and isinstance(key, str):
                new_key = redact(key)  # caller-shaped containers carry data in their keys too
                changed = changed or new_key != key
            new_item, item_changed = _walk(item, redact, full=full or key in CONTENT_KEYS)
            mapping[new_key] = new_item
            changed = changed or item_changed
        return mapping, changed
    return value, False


class ChannelRedactors:
    """Lazily created redactors keyed by channel."""

    def __init__(self, factory: RedactorFactory, max_channels: int = MAX_CHANNELS) -> None:
        self._factory = factory
        self._channels: dict[Hashable, StreamingRedactor] = {}
        self._closed_counts: Counter[str] = Counter()
        self._max_channels = max_channels
        self.overflowed = 0
        """Channels that were folded into the shared overflow channel (bounded memory)."""

    def get(self, key: Hashable, *, json: bool = False) -> StreamingRedactor:
        """The redactor for ``key``; ``json`` marks a channel that streams a JSON document.

        An upstream that keeps inventing channel keys (a fresh ``index`` per event)
        gets one shared channel past ``max_channels``: text is still redacted and
        memory stays bounded, at the price of unrelated fragments being scanned
        together (which can only add redactions, never lose one).
        """
        redactor = self._channels.get(key)
        if redactor is None:
            if len(self._channels) >= self._max_channels and key != _OVERFLOW_KEY:
                self.overflowed += 1
                return self.get(_OVERFLOW_KEY, json=json)
            redactor = self._factory(json_mode=True) if json else self._factory()
            self._channels[key] = redactor
        return redactor

    def flush_where(self, predicate: Callable[[Any], bool]) -> list[tuple[Any, str]]:
        """Flush and close every channel whose key satisfies ``predicate``."""
        flushed: list[tuple[Any, str]] = []
        for key in [k for k in self._channels if predicate(k)]:
            redactor = self._channels.pop(key)
            flushed.append((key, redactor.flush()))
            self._closed_counts.update(redactor.redactions)
        return flushed

    def flush_all(self) -> list[tuple[Any, str]]:
        return self.flush_where(lambda _key: True)

    @property
    def redaction_counts(self) -> Counter[str]:
        total = Counter(self._closed_counts)
        for redactor in self._channels.values():
            total.update(redactor.redactions)
        return total


class StreamAdapter(ABC):
    def __init__(self, redactor_factory: RedactorFactory) -> None:
        self.channels = ChannelRedactors(redactor_factory)
        self._redactor_factory = redactor_factory
        self._one_shot_counts: Counter[str] = Counter()
        self.stripped: Counter[str] = Counter()
        """Fields removed outright (currently only ``logprobs``), for the request log."""

    @property
    def redaction_counts(self) -> Counter[str]:
        """Redactions by detector, across streamed channels and one-shot passes."""
        return self.channels.redaction_counts + self._one_shot_counts

    # -- one-shot redaction --------------------------------------------------

    def redact_string(self, text: str) -> str:
        if not text:
            return text
        redactor = self._redactor_factory()
        result = redactor.feed(text) + redactor.flush()
        self._one_shot_counts.update(redactor.redactions)
        return result

    def redact_strings_in(self, value: Any) -> Any:
        """Redact every string inside a caller-shaped JSON structure, whatever its keys."""
        redacted, _ = _walk(value, self.redact_string, full=True)
        return redacted

    def redact_envelope(self, value: Any) -> Any:
        """Redact a provider envelope: opaque identifiers kept, content containers exhaustive."""
        redacted, _ = _walk(value, self.redact_string, full=False)
        return redacted

    def scrub(self, container: dict[str, Any], handled: Iterable[str] = ()) -> bool:
        """Redact in place every member of ``container`` except ``handled`` and opaque keys.

        Used for the parts of an event the adapter has no dedicated handling for.
        Returns whether anything changed, so untouched events can still be relayed
        byte for byte.
        """
        skip = set(handled)
        changed = False
        for key in list(container):
            if key in skip or key in OPAQUE_KEYS:
                continue
            new_value, item_changed = _walk(
                container[key], self.redact_string, full=key in CONTENT_KEYS
            )
            if item_changed:
                container[key] = new_value
                changed = True
        return changed

    def strip_logprobs(
        self, container: dict[str, Any], keys: tuple[str, ...] = ("logprobs",)
    ) -> bool:
        """Drop token-level log probabilities: they would leak the redacted text token by token."""
        stripped = False
        for key in keys:
            if container.get(key) is None:
                continue
            container[key] = None
            self.stripped[key] += 1
            stripped = True
        return stripped

    def redact_chunk_sequence(self, chunks: list[Any]) -> list[Any]:
        """Redact an array of streaming chunks delivered as one JSON document.

        Each element is run through ``process`` in order, exactly as it would
        have been as an SSE event, so PII split across two chunks is still
        caught; whatever is held back at the end is flushed through ``finish``.
        """
        parser = SseParser()
        blocks: list[bytes] = []
        for chunk in chunks:
            (event,) = parser.feed(serialize_event(dump_json(chunk)))
            blocks.extend(self.process(event))
        blocks.extend(self.finish())
        redacted: list[Any] = []
        for block in blocks:
            for event in parser.feed(block):
                if event.has_data:
                    redacted.append(parse_json(event.data))
        return redacted

    def redact_unparseable_event(self, event: SseEvent) -> bytes:
        """Fail closed for ``data`` that is not a JSON object: redact it as plain text."""
        return self._serialize(event, self.redact_string(event.data))

    def reserialize_fieldless(self, event: SseEvent) -> bytes:
        """Rebuild an event without ``data`` (comments, keep-alives, ids) with its text redacted."""
        return self._serialize(event, None)

    # -- serialisation -------------------------------------------------------

    def reserialize(self, event: SseEvent, payload: Any) -> bytes:
        return self._serialize(event, dump_json(payload))

    def _serialize(self, event: SseEvent, data: str | None) -> bytes:
        """Every field a client could read is redacted; nothing is relayed byte for byte."""
        return serialize_event(
            data,
            event=self.redact_string(event.event) if event.event is not None else None,
            event_id=self.redact_string(event.id) if event.id is not None else None,
            retry=event.retry,
            comments=tuple(self.redact_string(comment) for comment in event.comments),
        )

    # -- adapter contract ----------------------------------------------------

    @abstractmethod
    def process(self, event: SseEvent) -> list[bytes]:
        """Transform one upstream event into zero or more downstream byte blocks."""

    @abstractmethod
    def finish(self) -> list[bytes]:
        """Emit whatever is still held back when the upstream stream ends."""

    @abstractmethod
    def redact_response(self, payload: Any) -> Any:
        """Redact a complete (non-streaming) JSON response, an object or an array of them."""
