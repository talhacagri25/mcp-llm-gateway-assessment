"""Adapter for the OpenAI Chat Completions wire format.

Streaming: ``data: {chat.completion.chunk}`` events terminated by
``data: [DONE]``. Streamed text channels, each with its own chunk-boundary
safe redactor:

* ``choices[i].delta.content`` (assistant prose),
* ``choices[i].delta.refusal`` (refusal prose),
* ``choices[i].delta.reasoning_content`` and ``choices[i].delta.reasoning``
  (reasoning text exposed by some providers),
* ``choices[i].delta.audio.transcript`` (audio output transcripts),
* ``choices[i].delta.tool_calls[j].function.arguments`` and
  ``choices[i].delta.function_call.arguments`` (JSON documents: a bare number
  that turns out to be a card is replaced by a JSON string so the arguments
  stay parseable).

Held back text is appended to the chunk that carries ``finish_reason`` for
that choice, so the event count and ordering are preserved. If the stream
ends without a finish chunk, a synthetic chunk carries the remainder before
``[DONE]``. Everything else in a chunk (audio transcripts, annotations,
content filter results, fields added in the future, and any handled field
that arrives in an unexpected type) is redacted in one shot, and
``logprobs`` are removed.
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

DONE_MARKER = "[DONE]"

_TEXT_FIELDS: Final = ("content", "refusal", "reasoning_content", "reasoning")


class OpenAIChatAdapter(StreamAdapter):
    def __init__(self, redactor_factory: Any) -> None:
        super().__init__(redactor_factory)
        self._template: dict[str, Any] | None = None

    # -- streaming -----------------------------------------------------------

    def process(self, event: SseEvent) -> list[bytes]:
        if not event.has_data:
            return [self.reserialize_fieldless(event)]
        if event.data.strip() == DONE_MARKER:
            done = serialize_event(
                DONE_MARKER, event=event.event, event_id=event.id, retry=event.retry
            )
            return [*self._synthetic_flush_chunks(), done]
        payload = parse_json(event.data)
        if not isinstance(payload, dict):
            return [self.redact_unparseable_event(event)]

        consumed: set[str] = set()
        choices = payload.get("choices")
        if isinstance(choices, list):
            consumed.add("choices")
            for position, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    choices[position] = self.redact_strings_in(choice)
                    continue
                index = normalize_index(choice.get("index", 0))
                choice_consumed = {"logprobs"}
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    choice_consumed.add("delta")
                    self._feed_delta(delta, index)
                self.strip_logprobs(choice)
                self.scrub(choice, choice_consumed)
                if choice.get("finish_reason") is not None:
                    self._append_flush(choice, index)
        self.scrub(payload, consumed)

        self._remember_template(payload)
        return [self.reserialize(event, payload)]

    def _feed_delta(self, delta: dict[str, Any], index: int | str) -> None:
        consumed: set[str] = set()
        for field in _TEXT_FIELDS:
            value = delta.get(field)
            if isinstance(value, str):
                consumed.add(field)
                if value:
                    delta[field] = self.channels.get((field, index)).feed(value)

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            consumed.add("tool_calls")
            for position, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    tool_calls[position] = self.redact_strings_in(call)
                    continue
                call_consumed: set[str] = set()
                function = call.get("function")
                if isinstance(function, dict):
                    call_consumed.add("function")
                    function_consumed: set[str] = set()
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        function_consumed.add("arguments")
                        if arguments:
                            key = ("tool", index, normalize_index(call.get("index", 0)))
                            channel = self.channels.get(key, json=True)
                            function["arguments"] = channel.feed(arguments)
                    self.scrub(function, function_consumed)
                self.scrub(call, call_consumed)

        function_call = delta.get("function_call")
        if isinstance(function_call, dict):
            consumed.add("function_call")
            legacy_consumed: set[str] = set()
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                legacy_consumed.add("arguments")
                if arguments:
                    channel = self.channels.get(("function", index), json=True)
                    function_call["arguments"] = channel.feed(arguments)
            self.scrub(function_call, legacy_consumed)

        audio = delta.get("audio")
        if isinstance(audio, dict):
            consumed.add("audio")
            audio_consumed: set[str] = set()
            transcript = audio.get("transcript")
            if isinstance(transcript, str):
                audio_consumed.add("transcript")
                if transcript:
                    audio["transcript"] = self.channels.get(("audio", index)).feed(transcript)
            self.scrub(audio, audio_consumed)

        self.scrub(delta, consumed)

    def _append_flush(self, choice: dict[str, Any], index: int | str) -> None:
        """Attach held back text for ``index`` to the finishing chunk of that choice."""
        for key, text in self.channels.flush_where(lambda key: key[1] == index):
            if not text:
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
                choice["delta"] = delta
            self._place(delta, key, text)

    def _place(self, delta: dict[str, Any], key: Any, text: str) -> None:
        """Append flushed ``text`` to the delta field its channel ``key`` came from."""
        kind = key[0]
        if kind == "tool":
            self._append_tool_arguments(delta, key[2], text)
        elif kind == "function":
            function_call = delta.get("function_call")
            if not isinstance(function_call, dict):
                function_call = {}
                delta["function_call"] = function_call
            function_call["arguments"] = (function_call.get("arguments") or "") + text
        elif kind == "audio":
            audio = delta.get("audio")
            if not isinstance(audio, dict):
                audio = {}
                delta["audio"] = audio
            audio["transcript"] = (audio.get("transcript") or "") + text
        elif kind == "overflow":
            delta["content"] = (delta.get("content") or "") + text
        else:
            delta[kind] = (delta.get(kind) or "") + text

    @staticmethod
    def _append_tool_arguments(delta: dict[str, Any], call_index: Any, text: str) -> None:
        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
            delta["tool_calls"] = tool_calls
        for call in tool_calls:
            if isinstance(call, dict) and normalize_index(call.get("index", 0)) == call_index:
                function = call.setdefault("function", {})
                function["arguments"] = (function.get("arguments") or "") + text
                return
        tool_calls.append({"index": call_index, "function": {"arguments": text}})

    def _remember_template(self, payload: dict[str, Any]) -> None:
        if self._template is None:
            template = {k: v for k, v in payload.items() if k not in {"choices", "usage"}}
            template["object"] = payload.get("object", "chat.completion.chunk")
            self._template = template

    def _synthetic_flush_chunks(self) -> list[bytes]:
        """Chunks carrying text still held back when no finish chunk was seen."""
        flushed = [(key, text) for key, text in self.channels.flush_all() if text]
        if not flushed:
            return []
        choices: dict[Any, dict[str, Any]] = {}
        for key, text in flushed:
            index = key[1] if key[1] is not None else 0
            choice = choices.setdefault(index, {"index": index, "delta": {}, "finish_reason": None})
            self._place(choice["delta"], key, text)
        payload = copy.deepcopy(self._template or {"object": "chat.completion.chunk"})
        payload["choices"] = list(choices.values())
        return [serialize_event(dump_json(payload))]

    def finish(self) -> list[bytes]:
        return self._synthetic_flush_chunks()

    # -- non-streaming -------------------------------------------------------

    def redact_response(self, payload: Any) -> Any:
        if isinstance(payload, list):
            return self.redact_chunk_sequence(payload)
        if isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict):
                        self.strip_logprobs(choice)
        redacted: Any = self.redact_envelope(payload)
        return redacted
