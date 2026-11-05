"""Adapter for the Gemini ``generateContent`` wire format.

Streaming (``:streamGenerateContent?alt=sse``) delivers ``data:`` events whose
JSON carries ``candidates[i].content.parts[j]``; a part is ``{"text": ...}``
(with ``"thought": true`` for thinking output), a complete ``functionCall``
or inline binary data. Text parts of one candidate form one ordered stream per
kind (answer text and thought text are separate channels, so a token split
between a thought and the answer can never be joined into a false match).

Held back text is appended to the candidate's last part of that kind in the
chunk that carries ``finishReason``; if the stream ends without one, a
synthetic chunk carries the remainder. Function call arguments arrive whole
and are redacted exhaustively (every key, every string, every number); the
non-SSE streaming variant (a JSON array of chunks) and ``:generateContent``
responses go through the same envelope walk. Handled fields that arrive in an
unexpected type are redacted exhaustively rather than relayed.
"""

from __future__ import annotations

import copy
from typing import Any, Final

from llm_guardrail_gateway.adapters.base import (
    StreamAdapter,
    dump_json,
    normalize_index,
    parse_json,
)
from llm_guardrail_gateway.sse import SseEvent, serialize_event

_LOGPROB_KEYS: Final = ("logprobsResult",)


class GeminiGenerateContentAdapter(StreamAdapter):
    def __init__(self, redactor_factory: Any) -> None:
        super().__init__(redactor_factory)
        self._template: dict[str, Any] | None = None

    # -- streaming -----------------------------------------------------------

    def process(self, event: SseEvent) -> list[bytes]:
        if not event.has_data:
            return [self.reserialize_fieldless(event)]
        payload = parse_json(event.data)
        if not isinstance(payload, dict):
            return [self.redact_unparseable_event(event)]

        consumed: set[str] = set()
        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            consumed.add("candidates")
            for position, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    candidates[position] = self.redact_strings_in(candidate)
                    continue
                index = normalize_index(candidate.get("index", 0))
                candidate_consumed = set(_LOGPROB_KEYS)
                content = candidate.get("content")
                if isinstance(content, dict):
                    candidate_consumed.add("content")
                    self._feed_content(content, index)
                self.strip_logprobs(candidate, _LOGPROB_KEYS)
                self.scrub(candidate, candidate_consumed)
                if candidate.get("finishReason"):
                    self._append_flush(candidate, index)
        self.scrub(payload, consumed)

        self._remember_template(payload)
        return [self.reserialize(event, payload)]

    def _feed_content(self, content: dict[str, Any], index: int | str) -> None:
        consumed: set[str] = set()
        parts = content.get("parts")
        if isinstance(parts, list):
            consumed.add("parts")
            for position, part in enumerate(parts):
                if not isinstance(part, dict):
                    parts[position] = self.redact_strings_in(part)
                    continue
                part_consumed: set[str] = set()
                text = part.get("text")
                if isinstance(text, str):
                    part_consumed.add("text")
                    if text:
                        kind = "thought" if part.get("thought") else "text"
                        part["text"] = self.channels.get((kind, index)).feed(text)
                self.scrub(part, part_consumed)
        self.scrub(content, consumed)

    def _append_flush(self, candidate: dict[str, Any], index: int | str) -> None:
        for key, text in self.channels.flush_where(lambda key: key[1] == index):
            if text:
                self._place(candidate, key[0], text)

    @staticmethod
    def _place(candidate: dict[str, Any], kind: str, text: str) -> None:
        """Append flushed ``text`` to the last part of its kind, or add a part."""
        content = candidate.get("content")
        if not isinstance(content, dict):
            content = {"role": "model"}
            candidate["content"] = content
        parts = content.get("parts")
        if not isinstance(parts, list):
            parts = []
            content["parts"] = parts
        for part in reversed(parts):
            if (
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and bool(part.get("thought")) == (kind == "thought")
            ):
                part["text"] = part["text"] + text
                return
        new_part: dict[str, Any] = {"text": text}
        if kind == "thought":
            new_part["thought"] = True
        parts.append(new_part)

    def _remember_template(self, payload: dict[str, Any]) -> None:
        if self._template is None:
            self._template = {
                k: v for k, v in payload.items() if k not in {"candidates", "usageMetadata"}
            }

    def finish(self) -> list[bytes]:
        flushed = [(key, text) for key, text in self.channels.flush_all() if text]
        if not flushed:
            return []
        candidates: dict[Any, dict[str, Any]] = {}
        for key, text in flushed:
            kind, index = key[0], key[1]
            if kind == "overflow":
                kind, index = "text", 0
            candidate = candidates.setdefault(
                index, {"index": index, "content": {"parts": [], "role": "model"}}
            )
            self._place(candidate, kind, text)
        payload = copy.deepcopy(self._template or {})
        payload["candidates"] = list(candidates.values())
        return [serialize_event(dump_json(payload))]

    # -- non-streaming -------------------------------------------------------

    def redact_response(self, payload: Any) -> Any:
        if isinstance(payload, list):
            # The non-SSE streaming shape: chunks whose text continues across elements.
            return self.redact_chunk_sequence(payload)
        if isinstance(payload, dict):
            candidates = payload.get("candidates")
            if isinstance(candidates, list):
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        self.strip_logprobs(candidate, _LOGPROB_KEYS)
        redacted: Any = self.redact_envelope(payload)
        return redacted
