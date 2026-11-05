"""Provider adapters over synthetic SSE streams."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from llm_guardrail_gateway.adapters import AnthropicMessagesAdapter, OpenAIChatAdapter
from llm_guardrail_gateway.redaction.streaming import StreamingRedactor
from llm_guardrail_gateway.sse import SseEvent, SseParser, serialize_event


def _factory(*, json_mode: bool = False) -> StreamingRedactor:
    return StreamingRedactor(json_mode=json_mode)


def _event(payload: Any, event: str | None = None) -> SseEvent:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    (parsed,) = SseParser().feed(serialize_event(data, event=event))
    return parsed


def _chunk(
    content: str | None = None, finish: str | None = None, **delta_extra: Any
) -> dict[str, Any]:
    delta: dict[str, Any] = dict(delta_extra)
    if content is not None:
        delta["content"] = content
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _decode(blocks: list[bytes], sse_events: Callable[[bytes], list[SseEvent]]) -> list[SseEvent]:
    return sse_events(b"".join(blocks))


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_openai_text_split_across_chunks(sse_events: Callable[[bytes], list[SseEvent]]) -> None:
    adapter = OpenAIChatAdapter(_factory)
    out: list[bytes] = []
    for piece in ["Mail jo", "hn@exa", "mple.com ", "soon"]:
        out += adapter.process(_event(_chunk(piece)))
    out += adapter.process(_event(_chunk(finish="stop")))
    out += adapter.process(_event("[DONE]"))

    events = _decode(out, sse_events)
    assert events[-1].data == "[DONE]"
    contents = [json.loads(e.data)["choices"][0]["delta"].get("content", "") for e in events[:-1]]
    assert "".join(contents) == "Mail [REDACTED] soon"
    # The pending word rides on the finishing chunk, so event count is unchanged.
    assert len(events) == 6
    finish = json.loads(events[-2].data)["choices"][0]
    assert finish["finish_reason"] == "stop"
    assert finish["delta"]["content"] == "soon"
    assert adapter.redaction_counts == {"email": 1}


def test_openai_done_without_finish_emits_synthetic_chunk(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    out = adapter.process(_event(_chunk("ping a@b.io")))
    out += adapter.process(_event("[DONE]"))
    events = _decode(out, sse_events)
    assert [e.data for e in events][-1] == "[DONE]"
    synthetic = json.loads(events[-2].data)
    assert synthetic["object"] == "chat.completion.chunk"
    assert synthetic["id"] == "chatcmpl-1"
    assert synthetic["choices"][0]["delta"]["content"] == "[REDACTED]"
    assert synthetic["choices"][0]["finish_reason"] is None


def test_openai_stream_end_without_done_flushes(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    out = adapter.process(_event(_chunk("ssn 536-22-8041")))
    out += adapter.finish()
    events = _decode(out, sse_events)
    assert json.loads(events[0].data)["choices"][0]["delta"]["content"] == "ssn "
    assert json.loads(events[1].data)["choices"][0]["delta"]["content"] == "[REDACTED]"


def test_openai_passthrough_of_non_text_events(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    keepalive = _event(": keep-alive")
    (keepalive,) = SseParser().feed(b": keep-alive\n\n")
    role = _event(_chunk(role="assistant"))
    usage = _event(
        {"id": "x", "object": "chat.completion.chunk", "choices": [], "usage": {"total_tokens": 3}}
    )
    garbage = _event("not json at all")
    for original in (keepalive, role, usage, garbage):
        # Events are rebuilt from their fields (never relayed byte for byte), but
        # nothing about their content changes.
        (rebuilt,) = sse_events(b"".join(adapter.process(original)))
        assert rebuilt.has_data == original.has_data
        assert rebuilt.comments == original.comments
        if original.has_data:
            try:
                assert json.loads(rebuilt.data) == json.loads(original.data)
            except ValueError:
                assert rebuilt.data == original.data


def test_openai_choices_are_independent_channels(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)

    def two(c0: str, c1: str, finish: str | None = None) -> dict[str, Any]:
        return {
            "id": "c",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": c0}, "finish_reason": finish},
                {"index": 1, "delta": {"content": c1}, "finish_reason": finish},
            ],
        }

    out = adapter.process(_event(two("a@b", "536-22")))
    out += adapter.process(_event(two(".io ", "-8041 ")))
    out += adapter.process(_event(two("", "", "stop")))
    texts = {0: "", 1: ""}
    for event in _decode(out, sse_events):
        for choice in json.loads(event.data)["choices"]:
            texts[choice["index"]] += choice["delta"].get("content", "")
    assert texts == {0: "[REDACTED] ", 1: "[REDACTED] "}


def test_openai_tool_call_arguments_are_a_channel(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)

    def call(args: str, finish: str | None = None) -> dict[str, Any]:
        return _chunk(finish=finish, tool_calls=[{"index": 0, "function": {"arguments": args}}])

    out: list[bytes] = []
    for piece in ['{"to": "ceo@exa', 'mple.com", "n": 1}']:
        out += adapter.process(_event(call(piece)))
    out += adapter.process(_event(_chunk(finish="tool_calls")))
    arguments = ""
    for event in _decode(out, sse_events):
        for choice in json.loads(event.data)["choices"]:
            for item in choice["delta"].get("tool_calls", []):
                arguments += item["function"].get("arguments", "")
    assert json.loads(arguments) == {"to": "[REDACTED]", "n": 1}


def test_openai_non_streaming_response() -> None:
    adapter = OpenAIChatAdapter(_factory)
    payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "mail a@b.io"}},
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ssn 536-22-8041"}],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "f", "arguments": '{"card": "4111 1111 1111 1111"}'}}
                    ],
                }
            },
        ]
    }
    redacted = adapter.redact_response(payload)
    assert redacted["choices"][0]["message"]["content"] == "mail [REDACTED]"
    assert redacted["choices"][1]["message"]["content"][0]["text"] == "ssn [REDACTED]"
    assert (
        redacted["choices"][2]["message"]["tool_calls"][0]["function"]["arguments"]
        == '{"card": "[REDACTED]"}'
    )
    assert adapter.redact_response({"unexpected": True}) == {"unexpected": True}


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _delta(index: int, text: str) -> SseEvent:
    return _event(
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
        event="content_block_delta",
    )


def test_anthropic_text_blocks_flush_before_block_stop(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = AnthropicMessagesAdapter(_factory)
    out: list[bytes] = []
    start = _event({"type": "message_start", "message": {"id": "m"}}, event="message_start")
    ping = _event({"type": "ping"}, event="ping")
    for passthrough in (start, ping):
        (rebuilt,) = sse_events(b"".join(adapter.process(passthrough)))
        assert (rebuilt.event, json.loads(rebuilt.data)) == (
            passthrough.event,
            json.loads(passthrough.data),
        )
    for piece in ["card 4111 1111 ", "1111 1111", " done"]:
        out += adapter.process(_delta(0, piece))
    stop = _event({"type": "content_block_stop", "index": 0}, event="content_block_stop")
    out += adapter.process(stop)
    out += adapter.process(_event({"type": "message_stop"}, event="message_stop"))

    events = _decode(out, sse_events)
    kinds = [json.loads(e.data)["type"] for e in events]
    assert kinds == ["content_block_delta"] * 4 + ["content_block_stop", "message_stop"]
    assert all(e.event == json.loads(e.data)["type"] for e in events)
    text = "".join(json.loads(e.data)["delta"]["text"] for e in events[:4])
    assert text == "card [REDACTED] done"


def test_anthropic_blocks_are_independent_and_tool_input_is_redacted(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = AnthropicMessagesAdapter(_factory)
    out = adapter.process(_delta(0, "a@b"))
    out += adapter.process(
        _event(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"ssn": "536-22'},
            },
            event="content_block_delta",
        )
    )
    out += adapter.process(_delta(0, ".io!"))
    out += adapter.process(
        _event(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '-8041"}'},
            },
            event="content_block_delta",
        )
    )
    out += adapter.process(
        _event({"type": "content_block_stop", "index": 1}, event="content_block_stop")
    )
    out += adapter.process(
        _event({"type": "content_block_stop", "index": 0}, event="content_block_stop")
    )
    texts: dict[int, str] = {0: "", 1: ""}
    for event in _decode(out, sse_events):
        payload = json.loads(event.data)
        if payload["type"] == "content_block_delta":
            delta = payload["delta"]
            texts[payload["index"]] += delta.get("text") or delta.get("partial_json") or ""
    assert texts[0] == "[REDACTED]!"
    assert json.loads(texts[1]) == {"ssn": "[REDACTED]"}


def test_anthropic_non_streaming_response() -> None:
    adapter = AnthropicMessagesAdapter(_factory)
    payload = {
        "content": [
            {"type": "text", "text": "mail a@b.io"},
            {"type": "tool_use", "name": "f", "input": {"nested": {"ssn": "536-22-8041"}, "n": 2}},
        ]
    }
    redacted = adapter.redact_response(payload)
    assert redacted["content"][0]["text"] == "mail [REDACTED]"
    assert redacted["content"][1]["input"] == {"nested": {"ssn": "[REDACTED]"}, "n": 2}


# ---------------------------------------------------------------------------
# Fail-closed handling of everything outside the primary text channels
# ---------------------------------------------------------------------------


def test_openai_refusal_reasoning_and_legacy_function_call_are_channels(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    out: list[bytes] = []
    out += adapter.process(_event(_chunk(refusal="write to a@b", reasoning_content="ssn 536-22")))
    out += adapter.process(_event(_chunk(refusal=".io now", reasoning_content="-8041 ok")))
    out += adapter.process(
        _event(_chunk(function_call={"name": "f", "arguments": '{"c": "4111 1111'}))
    )
    out += adapter.process(_event(_chunk(function_call={"arguments": ' 1111 1111"}'})))
    out += adapter.process(_event(_chunk(finish="stop")))
    refusal = reasoning = arguments = ""
    for event in _decode(out, sse_events):
        delta = json.loads(event.data)["choices"][0]["delta"]
        refusal += delta.get("refusal", "")
        reasoning += delta.get("reasoning_content", "")
        arguments += (delta.get("function_call") or {}).get("arguments", "")
    assert refusal == "write to [REDACTED] now"
    assert reasoning == "ssn [REDACTED] ok"
    assert json.loads(arguments) == {"c": "[REDACTED]"}
    assert adapter.redaction_counts == {"email": 1, "ssn": 1, "credit_card": 1}


def test_openai_logprobs_are_stripped_from_chunks_and_responses(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    chunk = _chunk("hi ")
    chunk["choices"][0]["logprobs"] = {"content": [{"token": "a@b.io", "logprob": -0.1}]}
    (event,) = _decode(adapter.process(_event(chunk)), sse_events)
    assert json.loads(event.data)["choices"][0]["logprobs"] is None
    assert adapter.stripped == {"logprobs": 1}

    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "logprobs": {"content": [{"token": "536-22-8041"}]},
            }
        ]
    }
    assert adapter.redact_response(response)["choices"][0]["logprobs"] is None
    assert "536-22-8041" not in json.dumps(response)


def test_openai_unknown_string_fields_are_redacted_but_identifiers_kept(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    chunk = _chunk(
        audio={"id": "audio_4111111111111111", "transcript": "card 4111 1111 1111 1111 said "}
    )
    chunk["id"] = "chatcmpl-4111111111111111"
    chunk["choices"][0]["content_filter_results"] = {"note": "mail a@b.io"}
    (event,) = _decode(adapter.process(_event(chunk)), sse_events)
    payload = json.loads(event.data)
    assert payload["id"] == "chatcmpl-4111111111111111"
    choice = payload["choices"][0]
    assert choice["delta"]["audio"] == {
        "id": "audio_4111111111111111",
        "transcript": "card [REDACTED] said ",
    }
    assert choice["content_filter_results"] == {"note": "mail [REDACTED]"}
    assert adapter.redaction_counts == {"credit_card": 1, "email": 1}


def test_non_json_data_events_are_redacted_rather_than_relayed() -> None:
    for adapter in (OpenAIChatAdapter(_factory), AnthropicMessagesAdapter(_factory)):
        clean = _event("not json at all")
        assert adapter.process(clean) == [clean.raw]
        dirty = _event("reach me at a@b.io please", event="odd")
        (block,) = adapter.process(dirty)
        (parsed,) = SseParser().feed(block)
        assert parsed.event == "odd"
        assert parsed.data == "reach me at [REDACTED] please"
        (as_string,) = adapter.process(_event('"536-22-8041"'))
        assert as_string == b'data: "[REDACTED]"\n\n'


def test_openai_non_streaming_refusal_audio_and_annotations() -> None:
    adapter = OpenAIChatAdapter(_factory)
    payload = {
        "id": "chatcmpl-1",
        "model": "m",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "no, a@b.io"}],
                    "refusal": "no, a@b.io",
                    "audio": {"id": "a1", "data": "QUJD", "transcript": "ssn 536-22-8041"},
                    "annotations": [{"type": "url_citation", "url_citation": {"title": "c@d.io"}}],
                }
            }
        ],
        "usage": {"total_tokens": 3},
    }
    redacted = adapter.redact_response(payload)
    message = redacted["choices"][0]["message"]
    assert message["content"] == [{"type": "refusal", "refusal": "no, [REDACTED]"}]
    assert message["refusal"] == "no, [REDACTED]"
    assert message["audio"] == {"id": "a1", "data": "QUJD", "transcript": "ssn [REDACTED]"}
    assert message["annotations"][0]["url_citation"]["title"] == "[REDACTED]"
    assert redacted["usage"] == {"total_tokens": 3}
    assert adapter.redaction_counts == {"email": 3, "ssn": 1}


def test_anthropic_thinking_is_a_channel_flushed_before_block_stop(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = AnthropicMessagesAdapter(_factory)

    def thinking(piece: str) -> SseEvent:
        return _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": piece},
            },
            event="content_block_delta",
        )

    out: list[bytes] = []
    for piece in ["The user gave 536-", "22-8041 as", " their SSN"]:
        out += adapter.process(thinking(piece))
    signature = _event(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "4111111111111111=="},
        },
        event="content_block_delta",
    )
    # The held back thinking text is released before the signature that covers it.
    flushed_then_signature = adapter.process(signature)
    (rebuilt_signature,) = sse_events(flushed_then_signature[-1])
    assert json.loads(rebuilt_signature.data) == json.loads(signature.data)
    out += flushed_then_signature
    out += adapter.process(
        _event({"type": "content_block_stop", "index": 0}, event="content_block_stop")
    )
    events = _decode(out, sse_events)
    deltas = [json.loads(e.data)["delta"] for e in events[:-1]]
    assert [d["type"] for d in deltas][-1] == "signature_delta"
    thinking = [d for d in deltas if d["type"] == "thinking_delta"]
    assert "".join(d["thinking"] for d in thinking) == "The user gave [REDACTED] as their SSN"
    assert json.loads(events[-1].data)["type"] == "content_block_stop"


def test_anthropic_citations_and_prefilled_blocks_are_redacted(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = AnthropicMessagesAdapter(_factory)
    start = _event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "prefilled a@b.io"},
        },
        event="content_block_start",
    )
    citation = _event(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "char_location",
                    "cited_text": "ssn 536-22-8041",
                    "document_index": 0,
                },
            },
        },
        event="content_block_delta",
    )
    error = _event(
        {"type": "error", "error": {"type": "overloaded_error", "message": "retry c@d.io"}},
        event="error",
    )
    tool_start = _event(
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_4111111111111111",
                "name": "f",
                "input": {},
            },
        },
        event="content_block_start",
    )
    (rebuilt_tool_start,) = sse_events(b"".join(adapter.process(tool_start)))
    assert json.loads(rebuilt_tool_start.data) == json.loads(tool_start.data)
    stop = _event({"type": "content_block_stop", "index": 0}, event="content_block_stop")
    events = _decode(
        adapter.process(start)
        + adapter.process(citation)
        + adapter.process(error)
        + adapter.process(stop),
        sse_events,
    )
    # Pre-filled text is part of the block's stream: its tail is held like any delta.
    prefilled = json.loads(events[0].data)["content_block"]["text"]
    flushed = [
        json.loads(e.data)["delta"]["text"]
        for e in events
        if json.loads(e.data).get("delta", {}).get("type") == "text_delta"
    ]
    assert prefilled + "".join(flushed) == "prefilled [REDACTED]"
    assert json.loads(events[1].data)["delta"]["citation"]["cited_text"] == "ssn [REDACTED]"
    assert json.loads(events[2].data)["error"]["message"] == "retry [REDACTED]"
    assert events[0].event == "content_block_start"
    assert events[-1].event == "content_block_stop"


def test_anthropic_non_streaming_thinking_blocks_and_opaque_fields() -> None:
    adapter = AnthropicMessagesAdapter(_factory)
    payload = {
        "id": "msg_4111111111111111",
        "model": "m",
        "content": [
            {"type": "thinking", "thinking": "they said a@b.io", "signature": "4111111111111111"},
            {"type": "redacted_thinking", "data": "4111111111111111"},
            {"type": "text", "text": "ok", "citations": [{"cited_text": "536-22-8041"}]},
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"id": "536-22-8041"}},
        ],
        "stop_sequence": "end c@d.io",
    }
    redacted = adapter.redact_response(payload)
    assert redacted["id"] == "msg_4111111111111111"
    blocks = redacted["content"]
    assert blocks[0] == {
        "type": "thinking",
        "thinking": "they said [REDACTED]",
        "signature": "4111111111111111",
    }
    assert blocks[1] == {"type": "redacted_thinking", "data": "4111111111111111"}
    assert blocks[2]["citations"] == [{"cited_text": "[REDACTED]"}]
    assert blocks[3]["input"] == {"id": "[REDACTED]"}  # tool input: every key is caller data
    assert redacted["stop_sequence"] == "end [REDACTED]"


def test_redaction_counts_cover_one_shot_passes() -> None:
    adapter = OpenAIChatAdapter(_factory)
    adapter.redact_response({"choices": [{"message": {"content": "a@b.io and 536-22-8041"}}]})
    assert adapter.redaction_counts == {"email": 1, "ssn": 1}


def test_openai_audio_transcript_is_a_streamed_channel(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    out: list[bytes] = []
    for piece in ["mail jo", "hn@example.com now"]:
        out += adapter.process(_event(_chunk(audio={"id": "a1", "transcript": piece})))
    out += adapter.process(_event(_chunk(finish="stop")))
    transcript = ""
    for event in _decode(out, sse_events):
        transcript += (
            json.loads(event.data)["choices"][0]["delta"].get("audio", {}).get("transcript", "")
        )
    assert transcript == "mail [REDACTED] now"


def test_streamed_tool_arguments_stay_valid_json_for_bare_numbers(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    document = '{"card_number": 4111111111111111, "note": "ssn 536-22-8041", "n": 1}'
    out: list[bytes] = []
    for index in range(0, len(document), 7):
        piece = document[index : index + 7]
        out += adapter.process(
            _event(_chunk(tool_calls=[{"index": 0, "function": {"arguments": piece}}]))
        )
    out += adapter.process(_event(_chunk(finish="tool_calls")))
    arguments = ""
    for event in _decode(out, sse_events):
        for item in json.loads(event.data)["choices"][0]["delta"].get("tool_calls", []):
            arguments += item["function"].get("arguments", "")
    assert json.loads(arguments) == {
        "card_number": "[REDACTED]",
        "note": "ssn [REDACTED]",
        "n": 1,
    }


def test_anthropic_thinking_is_complete_before_its_signature(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = AnthropicMessagesAdapter(_factory)

    def delta(body: dict[str, Any]) -> SseEvent:
        return _event(
            {"type": "content_block_delta", "index": 0, "delta": body}, event="content_block_delta"
        )

    out = adapter.process(delta({"type": "thinking_delta", "thinking": "mail a@b"}))
    out += adapter.process(delta({"type": "thinking_delta", "thinking": ".io today"}))
    out += adapter.process(delta({"type": "signature_delta", "signature": "c2ln"}))
    out += adapter.process(
        _event({"type": "content_block_stop", "index": 0}, event="content_block_stop")
    )
    kinds = [json.loads(e.data).get("delta", {}).get("type") for e in _decode(out, sse_events)]
    assert kinds == ["thinking_delta", "thinking_delta", "thinking_delta", "signature_delta", None]
    thinking = "".join(
        json.loads(e.data)["delta"].get("thinking", "")
        for e in _decode(out, sse_events)
        if json.loads(e.data).get("type") == "content_block_delta"
    )
    assert thinking == "mail [REDACTED] today"


def test_numbers_in_tool_input_are_redacted_in_non_streaming_bodies() -> None:
    adapter = AnthropicMessagesAdapter(_factory)
    payload = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "pay",
                "input": {"card": 4111111111111111, "amount": 1999, "ok": True},
            }
        ]
    }
    redacted = adapter.redact_response(payload)
    assert redacted["content"][0]["input"] == {"card": "[REDACTED]", "amount": 1999, "ok": True}


def test_lone_surrogate_in_upstream_text_does_not_break_the_event() -> None:
    adapter = OpenAIChatAdapter(_factory)
    event = _event('{"choices":[{"index":0,"delta":{"content":"odd \\ud83d a@b.io "}}]}')
    (block,) = adapter.process(event)
    (parsed,) = SseParser().feed(block)
    assert "[REDACTED]" in parsed.data


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

from llm_guardrail_gateway.adapters import GeminiGenerateContentAdapter  # noqa: E402


def _gemini(parts: list[dict[str, Any]], finish: str | None = None, index: int = 0) -> SseEvent:
    candidate: dict[str, Any] = {"content": {"parts": parts, "role": "model"}, "index": index}
    if finish:
        candidate["finishReason"] = finish
    return _event({"candidates": [candidate], "modelVersion": "gemini-test"})


def _gemini_text(blocks: list[bytes], sse_events: Callable[[bytes], list[SseEvent]]) -> str:
    out = ""
    for event in _decode(blocks, sse_events):
        for candidate in json.loads(event.data).get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if not part.get("thought"):
                    out += part.get("text", "")
    return out


def test_gemini_text_split_across_events_is_flushed_on_finish(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = GeminiGenerateContentAdapter(_factory)
    out: list[bytes] = []
    for piece in ["Write to jo", "hn@exa", "mple.com or 536-22"]:
        out += adapter.process(_gemini([{"text": piece}]))
    out += adapter.process(_gemini([{"text": "-8041 today"}], finish="STOP"))
    events = _decode(out, sse_events)
    assert len(events) == 4  # the held text rides on the finishing chunk
    assert _gemini_text(out, sse_events) == "Write to [REDACTED] or [REDACTED] today"
    last = json.loads(events[-1].data)["candidates"][0]
    assert last["finishReason"] == "STOP"
    assert last["content"]["parts"][-1]["text"].endswith("today")
    assert adapter.redaction_counts == {"email": 1, "ssn": 1}


def test_gemini_thoughts_and_answers_are_separate_channels(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = GeminiGenerateContentAdapter(_factory)
    out = adapter.process(_gemini([{"text": "mail john@exa", "thought": True}]))
    out += adapter.process(_gemini([{"text": "mple.com"}], finish="STOP"))
    # Neither half is PII on its own, and the halves live in different channels,
    # so they are never joined into a false match. Held back text of each kind is
    # flushed into a part of the same kind.
    parts = [
        part
        for event in _decode(out, sse_events)
        for part in json.loads(event.data)["candidates"][0]["content"]["parts"]
    ]
    thought = "".join(p["text"] for p in parts if p.get("thought"))
    answer = "".join(p["text"] for p in parts if not p.get("thought"))
    assert (thought, answer) == ("mail john@exa", "mple.com")
    assert adapter.redaction_counts == {}


def test_gemini_function_call_arguments_are_redacted_whole(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = GeminiGenerateContentAdapter(_factory)
    call = {
        "functionCall": {
            "name": "pay",
            "args": {"card": 4111111111111111, "id": "536-22-8041", "nested": {"to": "a@b.io"}},
        }
    }
    (event,) = _decode(adapter.process(_gemini([call], finish="STOP")), sse_events)
    args = json.loads(event.data)["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]
    assert args == {"card": "[REDACTED]", "id": "[REDACTED]", "nested": {"to": "[REDACTED]"}}


def test_gemini_stream_without_finish_reason_flushes_at_the_end(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = GeminiGenerateContentAdapter(_factory)
    out = adapter.process(_gemini([{"text": "card 4111 1111 1111 1111"}]))
    out += adapter.finish()
    events = _decode(out, sse_events)
    assert json.loads(events[0].data)["modelVersion"] == "gemini-test"
    assert _gemini_text(out, sse_events) == "card [REDACTED]"
    synthetic = json.loads(events[-1].data)
    assert synthetic["modelVersion"] == "gemini-test"  # template carried over
    assert synthetic["candidates"][0]["content"]["role"] == "model"


def test_gemini_non_streaming_arrays_and_logprobs() -> None:
    adapter = GeminiGenerateContentAdapter(_factory)
    chunk = {
        "candidates": [
            {
                "content": {"parts": [{"text": "mail a@b.io"}], "role": "model"},
                "logprobsResult": {"topCandidates": [{"candidates": [{"token": "a@b.io"}]}]},
                "safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "LOW"}],
            }
        ],
        "usageMetadata": {"totalTokenCount": 3},
    }
    redacted = adapter.redact_response(chunk)
    assert redacted["candidates"][0]["content"]["parts"][0]["text"] == "mail [REDACTED]"
    assert redacted["candidates"][0]["logprobsResult"] is None
    assert redacted["candidates"][0]["safetyRatings"][0]["category"] == "HARM_CATEGORY_HATE_SPEECH"
    assert adapter.stripped == {"logprobsResult": 1}

    # An array of chunks is a stream in disguise: PII split across elements is caught.
    sequence = GeminiGenerateContentAdapter(_factory)
    chunks = [
        {
            "candidates": [
                {"content": {"parts": [{"text": "mail jo"}], "role": "model"}, "index": 0}
            ]
        },
        {"candidates": [{"content": {"parts": [{"text": "hn@example.com!"}]}, "index": 0}]},
        {
            "candidates": [
                {"content": {"parts": [{"text": " bye"}]}, "index": 0, "finishReason": "STOP"}
            ]
        },
    ]
    out = sequence.redact_chunk_sequence(chunks)
    text = "".join(
        part["text"]
        for element in out
        for candidate in element["candidates"]
        for part in candidate["content"]["parts"]
    )
    assert text == "mail [REDACTED]! bye"


# ---------------------------------------------------------------------------
# Fail closed on unexpected shapes, field smuggling and channel floods
# ---------------------------------------------------------------------------

CARD = "4111 1111 1111 1111"


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"index": 0, "delta": {"content": [CARD]}}]},
        {"choices": [{"index": 0, "delta": f"card {CARD}"}]},
        {"choices": {"0": {"delta": {"content": CARD}}}},
        {"choices": [CARD]},
        {"choices": [{"index": 0, "delta": {"tool_calls": {"function": {"arguments": CARD}}}}]},
        {
            "choices": [
                {"index": 0, "delta": {"tool_calls": [{"function": {"arguments": {"c": CARD}}}]}}
            ]
        },
        {"choices": [{"index": 0, "delta": {"audio": {"transcript": [CARD]}}}]},
        {"choices": [{"index": 0, "delta": {"function_call": CARD}}]},
    ],
)
def test_openai_handled_fields_in_unexpected_types_are_still_redacted(
    sse_events: Callable[[bytes], list[SseEvent]], payload: dict[str, Any]
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    (event,) = _decode(adapter.process(_event(payload)) + adapter.finish(), sse_events)
    assert CARD not in event.data
    assert "[REDACTED]" in event.data


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "content_block_delta", "index": 0, "delta": "ssn 536-22-8041"},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": ["536-22-8041"]},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": {"s": "536-22-8041"}},
        },
        {"type": "content_block_start", "index": 0, "content_block": "536-22-8041"},
        {
            "type": "message_start",
            "message": {"content": [{"type": "text", "text": "536-22-8041"}]},
        },
    ],
)
def test_anthropic_handled_fields_in_unexpected_types_are_still_redacted(
    sse_events: Callable[[bytes], list[SseEvent]], payload: dict[str, Any]
) -> None:
    adapter = AnthropicMessagesAdapter(_factory)
    (event,) = _decode(adapter.process(_event(payload)) + adapter.finish(), sse_events)
    assert "536-22-8041" not in event.data


@pytest.mark.parametrize(
    "payload",
    [
        {"candidates": [{"content": {"parts": ["mail a@b.io"]}}]},
        {"candidates": [{"content": {"parts": [{"text": ["mail a@b.io"]}]}}]},
        {"candidates": [{"content": "mail a@b.io"}]},
        {"candidates": {"0": {"content": {"parts": [{"text": "mail a@b.io"}]}}}},
        {
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "f", "args": "a@b.io"}}]}}
            ]
        },
    ],
)
def test_gemini_handled_fields_in_unexpected_types_are_still_redacted(
    sse_events: Callable[[bytes], list[SseEvent]], payload: dict[str, Any]
) -> None:
    adapter = GeminiGenerateContentAdapter(_factory)
    (event,) = _decode(adapter.process(_event(payload)) + adapter.finish(), sse_events)
    assert "a@b.io" not in event.data


def test_event_fields_and_comments_cannot_carry_text_past_the_redactor() -> None:
    adapter = OpenAIChatAdapter(_factory)
    raw = b": keepalive for 4111 1111 1111 1111\nid: 536-22-8041\nevent: a@b.io\nretry: 1000\n\n"
    (event,) = SseParser().feed(raw)
    (rebuilt,) = SseParser().feed(b"".join(adapter.process(event)))
    assert rebuilt.id == "[REDACTED]"
    assert rebuilt.event == "[REDACTED]"
    assert rebuilt.retry == "1000"
    assert rebuilt.comments == ("keepalive for [REDACTED]",)


def test_line_separator_characters_cannot_split_an_event_for_other_parsers() -> None:
    adapter = OpenAIChatAdapter(_factory)
    breaks = [chr(0x2028), chr(0x2029), chr(0x85), chr(0x1C), chr(0x1D), chr(0x1E)]
    content = "x" + breaks[0] + 'data: {"hidden": 1}' + breaks[1] + " y" + breaks[2] + " z"
    hidden = {"choices": [{"index": 0, "delta": {"content": content}}]}
    blocks = adapter.process(_event(hidden)) + adapter.finish()
    text = b"".join(blocks).decode()
    for char in breaks:
        assert char not in text
    rebuilt = SseParser().feed(b"".join(blocks))
    assert all(e.data.count("\n") == 0 for e in rebuilt)  # one data line per event
    streamed = "".join(
        json.loads(e.data)["choices"][0]["delta"].get("content", "") for e in rebuilt
    )
    assert streamed == content

    (fieldless,) = SseParser().feed(b"id: a" + breaks[0].encode() + b"data: smuggled\n\n")
    (out,) = adapter.process(fieldless)
    assert breaks[0].encode() not in out
    assert out.count(b"\n") == 2


def test_channel_count_is_bounded_per_response(
    sse_events: Callable[[bytes], list[SseEvent]],
) -> None:
    adapter = OpenAIChatAdapter(_factory)
    out: list[bytes] = []
    for n in range(500):
        chunk = {"choices": [{"index": f"c{n}", "delta": {"content": f"{CARD} "}}]}
        out += adapter.process(_event(chunk))
    out += adapter.finish()
    assert adapter.channels.overflowed > 0
    text = "".join(
        choice["delta"].get("content", "")
        for event in _decode(out, sse_events)
        for choice in json.loads(event.data)["choices"]
    )
    assert CARD not in text
    assert text.count("[REDACTED]") == 500


def test_tool_input_keys_floats_and_names_are_scanned() -> None:
    adapter = AnthropicMessagesAdapter(_factory)
    payload = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "card_4111111111111111",
                "input": {"john.doe@example.com": "x", "card": 4111111111111111.0, "n": 1.5},
            }
        ]
    }
    redacted = adapter.redact_response(payload)
    block = redacted["content"][0]
    assert block["name"] == "card_[REDACTED]"
    assert block["input"] == {"[REDACTED]": "x", "card": "[REDACTED]", "n": 1.5}
