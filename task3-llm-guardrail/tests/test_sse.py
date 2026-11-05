"""Incremental SSE parsing."""

from __future__ import annotations

import pytest

from llm_guardrail_gateway.sse import SseParseError, SseParser, serialize_event


def test_events_are_emitted_only_when_complete() -> None:
    parser = SseParser()
    assert parser.feed(b"data: hel") == []
    assert parser.feed(b"lo\n") == []
    (event,) = parser.feed(b"\n")
    assert event.data == "hello"
    assert event.has_data is True
    assert event.raw == b"data: hello\n\n"


def test_multiple_events_in_one_chunk_and_split_terminators() -> None:
    parser = SseParser()
    events = parser.feed(b"data: a\n\ndata: b\n\ndata: c\n")
    assert [e.data for e in events] == ["a", "b"]
    assert [e.data for e in parser.feed(b"\n")] == ["c"]


def test_crlf_and_multi_line_data_and_fields() -> None:
    parser = SseParser()
    raw = b"event: delta\r\nid: 7\r\nretry: 1000\r\ndata: line one\r\ndata: line two\r\n\r\n"
    (event,) = parser.feed(raw)
    assert event.event == "delta"
    assert event.id == "7"
    assert event.retry == "1000"
    assert event.data == "line one\nline two"
    assert event.raw == raw


def test_comments_and_keepalives_are_preserved_but_carry_no_data() -> None:
    parser = SseParser()
    (event,) = parser.feed(b": keep-alive\n\n")
    assert event.has_data is False
    assert event.data == ""
    assert event.raw == b": keep-alive\n\n"


def test_leading_space_is_stripped_once_only() -> None:
    parser = SseParser()
    (event,) = parser.feed(b"data:  two spaces\n\n")
    assert event.data == " two spaces"
    (event,) = parser.feed(b"data:no-space\n\n")
    assert event.data == "no-space"


def test_flush_returns_an_unterminated_trailing_event() -> None:
    parser = SseParser()
    assert parser.feed(b"data: tail") == []
    (event,) = parser.flush()
    assert event.data == "tail"
    assert parser.flush() == []


def test_flush_ignores_trailing_whitespace() -> None:
    parser = SseParser()
    parser.feed(b"\n")
    assert parser.flush() == []


def test_oversized_event_is_rejected() -> None:
    parser = SseParser(max_event_bytes=16)
    parser.feed(b"data: 0123456789")
    with pytest.raises(SseParseError):
        parser.feed(b"0123456789")


def test_multibyte_characters_split_across_chunks() -> None:
    parser = SseParser()
    raw = "data: café 日本\n\n".encode()
    assert parser.feed(raw[:9]) == []
    (event,) = parser.feed(raw[9:])
    assert event.data == "café 日本"


def test_serialize_round_trip() -> None:
    raw = serialize_event('{"a": 1}\nsecond', event="delta", event_id="3", retry="5")
    assert raw == b'event: delta\nid: 3\nretry: 5\ndata: {"a": 1}\ndata: second\n\n'
    (event,) = SseParser().feed(raw)
    assert event.event == "delta"
    assert event.data == '{"a": 1}\nsecond'


def test_leading_byte_order_mark_is_ignored() -> None:
    parser = SseParser()
    assert parser.feed(b"\xef") == []
    assert parser.feed(b"\xbb") == []
    (event,) = parser.feed(b"\xbfdata: first\n\n")
    assert event.has_data is True
    assert event.data == "first"
    # Only the very first mark is special; a later one is ordinary data.
    (later,) = parser.feed(b"data: \xef\xbb\xbfsecond\n\n")
    assert later.data == "\ufeffsecond"


@pytest.mark.parametrize(
    "raw",
    [
        b"data: a\n\r\ndata: b\n\r\n",
        b"data: a\r\n\rdata: b\r\n\r",
        b"data: a\r\n\ndata: b\r\n\n",
        b"data: a\n\rdata: b\n\r",
    ],
)
def test_mixed_line_endings_terminate_events(raw: bytes) -> None:
    parser = SseParser()
    events = parser.feed(raw)
    assert [e.data for e in events] == ["a", "b"]
    assert parser.flush() == []


def test_lone_surrogates_do_not_break_serialisation() -> None:
    raw = serialize_event("bad \ud83d char")
    assert raw.startswith(b"data: bad ")
    (event,) = SseParser().feed(raw)
    assert event.has_data
