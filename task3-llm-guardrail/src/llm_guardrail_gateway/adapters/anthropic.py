"""Adapter for the Anthropic Messages wire format.

Streaming events carry an ``event:`` name and a JSON ``data:`` body whose
``type`` is one of ``message_start``, ``content_block_start``,
``content_block_delta``, ``content_block_stop``, ``message_delta``,
``message_stop``, ``ping`` or ``error``. Streamed text channels, one per
content block index and delta kind:

* ``text_delta`` (assistant prose), including text pre-filled in the block's
  ``content_block_start``,
* ``thinking_delta`` (extended thinking, which can quote the prompt),
* ``input_json_delta`` (tool input).

Held back text is emitted as an extra delta of the same kind right before the
block's ``content_block_stop``, or, for thinking, right before the block's
``signature_delta`` so the thinking text is complete when its signature
arrives. ``signature_delta`` payloads are cryptographic material and pass
untouched; every other event (``citations_delta`` text, errors, fields added
in the future, handled fields in an unexpected type) is redacted in one shot.
Tool input streams in ``json_mode`` so a bare number that is a card becomes a
JSON string rather than invalid JSON.
"""

from __future__ import annotations

from typing import Any, Final

from llm_guardrail_gateway.adapters.base import (
    StreamAdapter,
    dump_json,
    normalize_index,
    parse_json,
)
from llm_guardrail_gateway.sse import SseEvent, serialize_event

_DELTA_CHANNELS: Final[dict[str, tuple[str, str]]] = {
    "text_delta": ("text", "text"),
    "input_json_delta": ("json", "partial_json"),
    "thinking_delta": ("thinking", "thinking"),
}
"""Delta ``type`` -> (channel kind, field holding the streamed text)."""

_BLOCK_CHANNELS: Final[dict[str, tuple[str, str]]] = {
    "text": ("text", "text"),
    "thinking": ("thinking", "thinking"),
}
"""``content_block_start`` block ``type`` -> (channel kind, pre-filled text field)."""

_FLUSH_SHAPES: Final[dict[str, tuple[str, str]]] = {
    kind: (delta_type, field) for delta_type, (kind, field) in _DELTA_CHANNELS.items()
}


class AnthropicMessagesAdapter(StreamAdapter):
    def process(self, event: SseEvent) -> list[bytes]:
        if not event.has_data:
            return [self.reserialize_fieldless(event)]
        payload = parse_json(event.data)
        if not isinstance(payload, dict):
            return [self.redact_unparseable_event(event)]

        kind = payload.get("type")
        preceding: list[bytes] = []
        if kind == "content_block_delta":
            preceding = self._process_delta(payload)
        elif kind == "content_block_start":
            self._process_block_start(payload)
        elif kind == "content_block_stop":
            index = normalize_index(payload.get("index", 0))
            preceding = self._delta_events(self.channels.flush_where(lambda key: key[1] == index))
            self.scrub(payload)
        elif kind == "message_stop":
            preceding = self._delta_events(self.channels.flush_all())
            self.scrub(payload)
        else:
            # message_start, message_delta, ping, error, unknown kinds.
            self.scrub(payload)
        return [*preceding, self.reserialize(event, payload)]

    def _process_delta(self, payload: dict[str, Any]) -> list[bytes]:
        index = normalize_index(payload.get("index", 0))
        delta = payload.get("delta")
        preceding: list[bytes] = []
        consumed: set[str] = set()
        if isinstance(delta, dict):
            consumed.add("delta")
            delta_consumed: set[str] = set()
            shape = _DELTA_CHANNELS.get(str(delta.get("type")))
            if shape is not None:
                channel, field = shape
                value = delta.get(field)
                if isinstance(value, str):
                    delta_consumed.add(field)
                    if value:
                        redactor = self.channels.get((channel, index), json=channel == "json")
                        delta[field] = redactor.feed(value)
            elif delta.get("type") == "signature_delta":
                # The signature covers the whole thinking text: release it first.
                flushed = self.channels.flush_where(lambda key: key == ("thinking", index))
                preceding = self._delta_events(flushed)
            self.scrub(delta, delta_consumed)
        self.scrub(payload, consumed)
        return preceding

    def _process_block_start(self, payload: dict[str, Any]) -> None:
        index = normalize_index(payload.get("index", 0))
        block = payload.get("content_block")
        consumed: set[str] = set()
        if isinstance(block, dict):
            consumed.add("content_block")
            block_consumed: set[str] = set()
            shape = _BLOCK_CHANNELS.get(str(block.get("type")))
            if shape is not None:
                channel, field = shape
                value = block.get(field)
                if isinstance(value, str):
                    block_consumed.add(field)
                    if value:
                        # Pre-filled text belongs to the same stream as the deltas that follow.
                        block[field] = self.channels.get((channel, index)).feed(value)
            self.scrub(block, block_consumed)
        self.scrub(payload, consumed)

    @staticmethod
    def _delta_events(flushed: list[tuple[Any, str]]) -> list[bytes]:
        events: list[bytes] = []
        for key, text in flushed:
            if not text:
                continue
            kind, index = key[0], key[1]
            if kind == "overflow":
                kind, index = "text", 0
            delta_type, field = _FLUSH_SHAPES[kind]
            payload = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": delta_type, field: text},
            }
            events.append(serialize_event(dump_json(payload), event="content_block_delta"))
        return events

    def finish(self) -> list[bytes]:
        return self._delta_events(self.channels.flush_all())

    def redact_response(self, payload: Any) -> Any:
        if isinstance(payload, list):
            return self.redact_chunk_sequence(payload)
        redacted: Any = self.redact_envelope(payload)
        return redacted
