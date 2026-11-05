"""Chunk boundary behaviour of the streaming redactor."""

from __future__ import annotations

import random

import pytest

from llm_guardrail_gateway.redaction.patterns import DEFAULT_PATTERNS, EMAIL, SSN
from llm_guardrail_gateway.redaction.streaming import StreamingRedactor, redact_text

CORPUS = [
    "Contact john.doe@example.com today.",
    "Sure. You can reach our billing lead at maria.santos@example.com or her backup "
    "ops-team+billing@mail.example.org. The customer's SSN on file is 536-22-8041 and the "
    "card used was 4111 1111 1111 1111 (expires 09/27). Order 1234567890 shipped.",
    "Card: 4111-1111-1111-1111 and amex 3782 822463 10005 and invalid 1234 5678 9012 3456.",
    "SSN 123-45-6789 vs invalid 000-12-3456; spaced 123 45 6789.",
    "x123-45-6789 edge, ID123-45-6789, (123-45-6789), 1123-45-6789 longer",
    "email at end user@example.com",
    "user@example.co then m continues",
    "a@b.cc12 34 and tail digits 4111 1111 1111 111",
    "nothing here at all. plain words only",
    "unicode: café résumé plain@example.com and 日本語 536-22-8041 text",
    "multi\nline\njane@example.org\n123-45-6789\n",
    "4111 1111 1111 1111.",
    "john@example.com123-45-6789",
    "numbers 12 34 56 78 90 12 34 56 78 90 end",
    "",
    "a",
    "123-45-",
    # Cases where a rejected digit run precedes or follows a real card number.
    "Numbers: 256553062-6147-57378 6011 1111 1111 1117 ok",
    "card then cvv 4111 1111 1111 1111 123 and 10045 4111 1111 1111 1111 end",
    "3782 822463 10005 1234 amex plus cvv",
    "a" * 257 + "@example.com local part too long",
    'tool {"card": 4111111111111111, "note": "ref: 536-22-8041"}',
    "nbsp card 4111\u00a01111\u00a01111\u00a01111 and ssn 536\u00a022\u00a08041",
]


def _stream(text: str, cuts: list[int], **kwargs: object) -> tuple[str, int]:
    redactor = StreamingRedactor(DEFAULT_PATTERNS, **kwargs)  # type: ignore[arg-type]
    pieces: list[str] = []
    previous = 0
    worst_pending = 0
    for cut in [*cuts, len(text)]:
        pieces.append(redactor.feed(text[previous:cut]))
        previous = cut
        worst_pending = max(worst_pending, len(redactor.pending))
    pieces.append(redactor.flush())
    return "".join(pieces), worst_pending


@pytest.mark.parametrize("text", CORPUS)
def test_every_single_split_point_matches_the_one_shot_result(text: str) -> None:
    expected = redact_text(text)
    for cut in range(len(text) + 1):
        assert _stream(text, [cut])[0] == expected, f"split at {cut}"


@pytest.mark.parametrize("text", CORPUS)
@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 11, 64])
def test_fixed_chunk_sizes_match_the_one_shot_result(text: str, size: int) -> None:
    cuts = list(range(size, len(text), size))
    assert _stream(text, cuts)[0] == redact_text(text)


@pytest.mark.parametrize("text", [t for t in CORPUS if len(t) > 8])
def test_random_multi_splits_match_the_one_shot_result(text: str) -> None:
    rnd = random.Random(1234)
    expected = redact_text(text)
    for _ in range(200):
        count = rnd.randint(1, min(10, len(text) - 1))
        cuts = sorted(rnd.sample(range(1, len(text)), count))
        assert _stream(text, cuts)[0] == expected, cuts


def test_prose_is_released_without_waiting_for_the_next_chunk() -> None:
    redactor = StreamingRedactor()
    assert redactor.feed("Hello ") == "Hello "  # trailing space: nothing can be pending
    assert redactor.feed("world") == ""  # a trailing word might be an email local part
    assert redactor.pending == "world"
    assert redactor.feed(", how are you? ") == "world, how are you? "
    assert redactor.feed("Fine") == ""
    assert redactor.flush() == "Fine"
    assert redactor.pending == ""


def test_pii_split_across_three_chunks_is_redacted_once() -> None:
    redactor = StreamingRedactor()
    out = redactor.feed("write to jo") + redactor.feed("hn@exa") + redactor.feed("mple.com now ")
    assert out == "write to [REDACTED] now "
    assert redactor.redactions == {"email": 1}


def test_holdback_is_bounded_even_for_pathological_input() -> None:
    longest = max(p.max_length for p in DEFAULT_PATTERNS)
    redactor = StreamingRedactor(max_holdback=32)
    released = redactor.feed("x" * 2000)
    assert len(redactor.pending) == longest  # the cap is raised to the longest possible match
    assert len(released) == 2000 - longest
    _, worst = _stream("1 " * 3000, list(range(7, 6000, 7)), max_holdback=32)
    assert worst <= 2 * longest


def test_flush_on_empty_and_repeated_flush_are_safe() -> None:
    redactor = StreamingRedactor()
    assert redactor.feed("") == ""
    assert redactor.flush() == ""
    assert redactor.feed("a@b.io") == ""
    assert redactor.flush() == "[REDACTED]"
    assert redactor.flush() == ""


def test_redaction_counts_are_tracked_per_pattern() -> None:
    text = "a@b.io and c@d.io and 536-22-8041 and 4111 1111 1111 1111"
    redactor = StreamingRedactor()
    redactor.feed(text)
    redactor.flush()
    assert redactor.redactions == {"email": 2, "ssn": 1, "credit_card": 1}


def test_pattern_subset_and_custom_replacement() -> None:
    redactor = StreamingRedactor([EMAIL], replacement="***")
    assert redactor.feed("a@b.io 536-22-8041 ") == "*** 536-22-8041 "
    redactor = StreamingRedactor([SSN])
    # With only the SSN detector active the email is plain text and released at once,
    # while the trailing number group waits for the next chunk or the flush.
    assert redactor.feed("a@b.io 536-22-8041 ") == "a@b.io "
    assert redactor.flush() == "[REDACTED] "


def test_lookbehind_context_survives_chunk_boundaries() -> None:
    # "1" + "123-45-6789": the digit before the SSN disqualifies it, even across a chunk cut.
    redactor = StreamingRedactor()
    out = redactor.feed("1") + redactor.feed("123-45-6789 ") + redactor.flush()
    assert out == "1123-45-6789 "
    redactor = StreamingRedactor()
    out = redactor.feed("(") + redactor.feed("123-45-6789") + redactor.flush()
    assert out == "([REDACTED]"


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_holdback"):
        StreamingRedactor(max_holdback=0)
    with pytest.raises(ValueError, match="pattern"):
        StreamingRedactor([])


def test_random_digit_runs_before_a_card_never_break_invariance() -> None:
    """A rejected (non-Luhn) digit run must not swallow the card that follows it."""
    rnd = random.Random(2024)
    for _ in range(300):
        prefix = "".join(rnd.choice("0123456789 -") for _ in range(rnd.randint(0, 30)))
        text = f"Numbers: {prefix} 6011 1111 1111 1117 ok"
        expected = redact_text(text)
        assert "6011 1111 1111 1117" not in expected
        for size in (1, 3, 7, 10):
            assert _stream(text, list(range(size, len(text), size)))[0] == expected, (text, size)


def test_scanning_is_linear_on_long_alphanumeric_runs() -> None:
    import time

    for text in ("a" * 80_000, "a" * 20_000 + "@" + "b" * 20_000 + "!"):
        started = time.perf_counter()
        redact_text(text)
        assert time.perf_counter() - started < 0.5


def test_long_email_runs_are_bounded_and_consistent() -> None:
    redactor = StreamingRedactor(DEFAULT_PATTERNS)
    redactor.feed("a" * 600)
    assert len(redactor.pending) <= EMAIL.max_length
    out = redactor.feed("@x.io") + redactor.flush()
    assert out.endswith("[REDACTED]")
    assert redact_text("a" * 600 + "@x.io") == "a" * 536 + "[REDACTED]"


def test_json_mode_keeps_tool_arguments_parseable() -> None:
    import json

    document = (
        '{"card_number": 4111111111111111, "note": "ref: 4111 1111 1111 1111", '
        '"ssn": 536228041, "mail": "a@b.io", "n": 1}'
    )
    expected = json.loads(redact_text(document, json_mode=True))
    assert expected == {
        "card_number": "[REDACTED]",
        "note": "ref: [REDACTED]",
        "ssn": 536228041,  # no separators: not an SSN by design
        "mail": "[REDACTED]",
        "n": 1,
    }
    for size in (1, 2, 5, 9):
        redactor = StreamingRedactor(DEFAULT_PATTERNS, json_mode=True)
        pieces = [redactor.feed(document[i : i + size]) for i in range(0, len(document), size)]
        pieces.append(redactor.flush())
        assert json.loads("".join(pieces)) == expected, size


def test_embedded_json_objects_in_prose_stay_parseable() -> None:
    import json

    text = (
        "Calling the tool now:\n<tool_call>\n"
        '{"name": "pay", "arguments": {"card": 4111111111111111, "note": "ssn 536-22-8041"}}\n'
        "</tool_call>\nDone, ref: 4111 1111 1111 1111, bye"
    )
    expected = redact_text(text)
    start, end = expected.index("{"), expected.rindex("}") + 1
    assert json.loads(expected[start:end]) == {
        "name": "pay",
        "arguments": {"card": "[REDACTED]", "note": "ssn [REDACTED]"},
    }
    # Outside the object the replacement is plain text again.
    assert expected.endswith("Done, ref: [REDACTED], bye")
    for size in (1, 2, 5, 9, 17):
        assert _stream(text, list(range(size, len(text), size)))[0] == expected, size


def test_prose_braces_that_are_not_json_do_not_add_quotes() -> None:
    text = "use {name} for 4111 1111 1111 1111 please, and {x: 1} too: 536-22-8041."
    assert redact_text(text) == "use {name} for [REDACTED] please, and {x: 1} too: [REDACTED]."


def test_whole_document_mode_quotes_a_bare_top_level_number() -> None:
    assert redact_text("4111111111111111", json_mode=True) == '"[REDACTED]"'
    assert redact_text("[4111111111111111, 2]", json_mode=True) == '["[REDACTED]", 2]'


# ---------------------------------------------------------------------------
# Representation evasions: the detection view
# ---------------------------------------------------------------------------

EVASIONS = {
    "zero width spaces": "pay 4111\u200b 1111\u200b 1111\u200b 1111 now",
    "joiner after every digit": (
        "pay 4\u200d1\u200d1\u200d1\u200d1\u200d1\u200d1\u200d1\u200d1"
        "\u200d1\u200d1\u200d1\u200d1\u200d1\u200d1\u200d1 now"
    ),
    "fullwidth digits": (
        "pay \uff14\uff11\uff11\uff11 \uff11\uff11\uff11\uff11 "
        "\uff11\uff11\uff11\uff11 \uff11\uff11\uff11\uff11 now"
    ),
    "fullwidth at sign": "mail john\uff20example\uff0ecom now",
    "zero width in address": "mail jo\u200bhn@exam\u200bple.com now",
    "en dash separators": "ssn 536\u201322\u20138041 ok",
    "minus sign separators": "ssn 536\u221222\u22128041 ok",
    "arabic indic digits": "ssn \u0665\u0663\u0666-\u0662\u0662-\u0668\u0660\u0664\u0661 ok",
    "combining mark": "card 4\u0301111 1111 1111 1111 ok",
    "figure spaces": "card 4111\u20071111\u20071111\u20071111 ok",
    "soft hyphen": "ssn 536\xad-22-8041 ok",
    "bidi override": "\u202e4111 1111 1111 1111\u202c ok",
    "nul inside": "card 41\x0011 1111 1111 1111 ok",
    "cyrillic domain": "john@ex\u0430mple.com",
    "superscript digits": "ssn \u2075\xb3\u2076-22-8041 ok",
    "markdown emphasis": "4111 1111 **1111 1111** end",
}


@pytest.mark.parametrize("text", list(EVASIONS.values()), ids=list(EVASIONS))
def test_representation_evasions_are_redacted_and_chunking_invariant(text: str) -> None:
    expected = redact_text(text)
    assert "[REDACTED]" in expected, expected
    assert "4111" not in expected
    assert "8041" not in expected
    assert "example" not in expected
    for size in (1, 2, 3, 5, 9):
        assert _stream(text, list(range(size, len(text), size)))[0] == expected, size


def test_adversarial_digit_runs_stay_cheap() -> None:
    import time

    for payload in ("1 2 " * 65536, "1 " * 131072, "4111 1111 1111 1111 " * 13000):
        started = time.perf_counter()
        redact_text(payload)
        assert time.perf_counter() - started < 1.0, len(payload)


def test_long_digit_runs_are_identifiers_not_cards() -> None:
    assert redact_text("4111111111111111" + "0" * 14) == "4111111111111111" + "0" * 14
    assert redact_text("41111111111111110") == "41111111111111110"  # no boundary inside the run
