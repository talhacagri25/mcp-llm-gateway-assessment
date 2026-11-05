"""Detector precision: what is and is not treated as PII."""

from __future__ import annotations

import pytest

from llm_guardrail_gateway.redaction.patterns import (
    CREDIT_CARD,
    EMAIL,
    SSN,
    PiiPattern,
    luhn_checksum_ok,
)
from llm_guardrail_gateway.redaction.streaming import redact_text


def _matches(pattern: PiiPattern, text: str) -> list[str]:
    return [m.group(0) for m in pattern.regex.finditer(text) if pattern.accepts(m.group(0))]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail john.doe@example.com now", ["john.doe@example.com"]),
        ("ops-team+billing@mail.example.org", ["ops-team+billing@mail.example.org"]),
        ("UPPER@EXAMPLE.IO", ["UPPER@EXAMPLE.IO"]),
        ("x@y.co.uk.", ["x@y.co.uk"]),
        ("a_b%c@sub.domain-name.museum", ["a_b%c@sub.domain-name.museum"]),
        ("two a@b.cc and c@d.dd", ["a@b.cc", "c@d.dd"]),
    ],
)
def test_email_matches(text: str, expected: list[str]) -> None:
    assert _matches(EMAIL, text) == expected


@pytest.mark.parametrize(
    "text", ["user@localhost", "not an email @ all", "a@b", "a@.com", "@example.com"]
)
def test_email_non_matches(text: str) -> None:
    assert _matches(EMAIL, text) == []


@pytest.mark.parametrize(
    "digits",
    [
        "4111111111111111",
        "378282246310005",
        "5555555555554444",
        "6011111111111117",
        "30569309025904",
    ],
)
def test_luhn_accepts_known_test_numbers(digits: str) -> None:
    assert luhn_checksum_ok(digits)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("card 4111 1111 1111 1111 ok", ["4111 1111 1111 1111"]),
        ("card 4111-1111-1111-1111 ok", ["4111-1111-1111-1111"]),
        ("card 4111111111111111 ok", ["4111111111111111"]),
        ("amex 3782 822463 10005.", ["3782 822463 10005"]),
        ("diners 30569309025904", ["30569309025904"]),
    ],
)
def test_card_matches(text: str, expected: list[str]) -> None:
    assert _matches(CREDIT_CARD, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "4111 1111 1111 1112",  # Luhn failure
        "1234 5678 9012 3456",  # Luhn failure
        "0000 0000 0000 0000",  # passes Luhn but is not a card number
        "order 123456789012",  # twelve digits
        "41111111111111111111",  # twenty digits
        "phone 555 0100 1234",
    ],
)
def test_card_non_matches(text: str) -> None:
    assert _matches(CREDIT_CARD, text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ssn 536-22-8041.", ["536-22-8041"]),
        ("ssn 536 22 8041", ["536 22 8041"]),
        ("ID536-22-8041", ["536-22-8041"]),
        ("(536-22-8041)", ["536-22-8041"]),
    ],
)
def test_ssn_matches(text: str, expected: list[str]) -> None:
    assert _matches(SSN, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "000-12-3456",
        "666-12-3456",
        "912-12-3456",
        "123-00-4567",
        "123-45-0000",
        "1536-22-8041",  # part of a longer number
        "536-22-80411",
        "536-228-041",
    ],
)
def test_ssn_non_matches(text: str) -> None:
    assert _matches(SSN, text) == []


def test_redact_text_replaces_every_kind_once() -> None:
    text = "Mail a@b.io, SSN 536-22-8041, card 4111 1111 1111 1111, order 1234567890."
    assert redact_text(text) == (
        "Mail [REDACTED], SSN [REDACTED], card [REDACTED], order 1234567890."
    )


def test_overlapping_candidates_do_not_double_redact() -> None:
    # The email ends right where an SSN-looking number starts; both are replaced exactly once.
    assert redact_text("john@example.com123-45-6789") == "[REDACTED][REDACTED]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4111 1111 1111 1111 123", "[REDACTED] 123"),  # card followed by its CVV
        ("4111 1111 1111 1111 1", "[REDACTED] 1"),
        ("3782 822463 10005 1234", "[REDACTED] 1234"),
        ("123 4111 1111 1111 1111", "123 [REDACTED]"),
        ("10045 4111 1111 1111 1111", "10045 [REDACTED]"),
        ("4111-1111-1111-1111-123", "[REDACTED]-123"),  # separators are all alike
        ("4111 1111 1111 1111-", "[REDACTED]-"),
        ("-4111 1111 1111 1111", "-[REDACTED]"),
        ("4111.1111.1111.1111", "[REDACTED]"),
        ("4111 1111 **1111 1111**", "[REDACTED]**"),
        ("4111\n1111\n1111\n1111", "[REDACTED]"),
        ("536.22.8041 and 536-22-8041-1", "[REDACTED] and [REDACTED]-1"),
        ("4111111111111111123", "4111111111111111123"),  # no boundary inside a digit run
        ("4111\u00a01111\u00a01111\u00a01111", "[REDACTED]"),  # non-breaking spaces
        ("ref 2024-1234-5678-9012-3456 end", "ref 2024-1234-5678-9012-3456 end"),
    ],
)
def test_card_resolution_inside_longer_digit_runs(text: str, expected: str) -> None:
    assert redact_text(text) == expected


def test_email_length_bounds_follow_the_rfc() -> None:
    assert redact_text("a" * 64 + "@x.io") == "[REDACTED]"
    assert redact_text("a" * 65 + "@x.io") == "a[REDACTED]"  # 64 character local part
    long_domain = "x@" + ".".join(["d" * 60] * 5) + ".com"  # over 254 characters
    assert len(long_domain) > 254
    # Not a deliverable address: left alone, consistently in both scan modes.
    assert redact_text(long_domain) == long_domain
    assert EMAIL.resolve(long_domain) == []


# ---------------------------------------------------------------------------
# Optional presets and declared detectors
# ---------------------------------------------------------------------------

from llm_guardrail_gateway.redaction.custom import (  # noqa: E402
    PatternConfigurationError,
    build_pattern,
    load_pattern_document,
    regex_alphabet,
)
from llm_guardrail_gateway.redaction.patterns import (  # noqa: E402
    DEFAULT_PATTERNS,
    IBAN,
    PATTERNS_BY_NAME,
    PHONE_E164,
    PHONE_US,
    SSN_COMPACT,
    iban_checksum_ok,
)
from llm_guardrail_gateway.redaction.streaming import StreamingRedactor  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("iban GB82 WEST 1234 5698 7654 32 ok", "iban [REDACTED] ok"),
        ("DE89370400440532013000", "[REDACTED]"),
        ("GB82 WEST 1234 5698 7654 33", "GB82 WEST 1234 5698 7654 33"),  # bad check digits
        ("ref ABCD1234 short", "ref ABCD1234 short"),
    ],
)
def test_iban_preset(text: str, expected: str) -> None:
    assert redact_text(text, (IBAN,)) == expected


def test_iban_checksum() -> None:
    assert iban_checksum_ok("GB82WEST12345698765432")
    assert not iban_checksum_ok("GB82WEST12345698765433")
    assert not iban_checksum_ok("1234567890123456")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call (415) 555-2671 now", "call [REDACTED] now"),
        ("call 415-555-2671", "call [REDACTED]"),
        ("call +1 415 555 2671", "call [REDACTED]"),
        ("call 415.555.2671", "call [REDACTED]"),
        ("123-456-7890 is invalid", "123-456-7890 is invalid"),  # area code cannot start with 1
        ("911-555-1234 is a service code", "911-555-1234 is a service code"),
    ],
)
def test_us_phone_preset(text: str, expected: str) -> None:
    assert redact_text(text, (PHONE_US,)) == expected


def test_e164_and_compact_ssn_presets_are_opt_in() -> None:
    assert redact_text("ring +14155552671 please", (PHONE_E164,)) == "ring [REDACTED] please"
    assert redact_text("ring +14155552671 please") == "ring +14155552671 please"
    assert redact_text("ssn 536228041 here", (SSN_COMPACT,)) == "ssn [REDACTED] here"
    assert redact_text("ssn 000123456 here", (SSN_COMPACT,)) == "ssn 000123456 here"
    assert redact_text("ssn 536228041 here") == "ssn 536228041 here"  # not a default detector
    assert set(DEFAULT_PATTERNS) < set(PATTERNS_BY_NAME.values())


def test_every_shipped_detector_declares_a_complete_alphabet() -> None:
    for pattern in PATTERNS_BY_NAME.values():
        consumed = regex_alphabet(pattern.regex)
        assert consumed is not None, pattern.name
        assert consumed <= pattern.holdback_chars, (pattern.name, consumed - pattern.holdback_chars)
        assert pattern.start_chars <= pattern.holdback_chars, pattern.name


def test_declared_detector_gets_the_streaming_guarantee() -> None:
    customer_id = build_pattern(
        {
            "name": "customer_id",
            "regex": "CUST-[0-9]{5}",
            "alphabet": "CUST-0123456789",
            "start_chars": "C",
            "max_length": 10,
        }
    )
    text = "ids CUST-00042 and CUST-1234 (short) and CUST-99999. done"
    patterns = (*DEFAULT_PATTERNS, customer_id)
    expected = redact_text(text, patterns)
    assert expected == "ids [REDACTED] and CUST-1234 (short) and [REDACTED]. done"
    for cut in range(len(text) + 1):
        redactor = StreamingRedactor(patterns)
        assert redactor.feed(text[:cut]) + redactor.feed(text[cut:]) + redactor.flush() == expected


def test_declared_detector_with_luhn_validator() -> None:
    account = build_pattern(
        {
            "name": "account",
            "regex": "(?<![0-9])[0-9]{10,12}(?![0-9])",
            "alphabet": "0123456789",
            "max_length": 12,
            "validator": "luhn",
        }
    )
    assert redact_text("acct 79927398713 ok", (account,)) == "acct [REDACTED] ok"
    assert redact_text("acct 79927398710 ok", (account,)) == "acct 79927398710 ok"


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"name": "Bad Name", "regex": "x", "alphabet": "x", "max_length": 1}, "name must match"),
        ({"name": "email", "regex": "x", "alphabet": "x", "max_length": 1}, "shadows"),
        ({"name": "a", "regex": "(", "alphabet": "x", "max_length": 1}, "invalid regex"),
        ({"name": "a", "regex": "x", "max_length": 1}, "alphabet is required"),
        ({"name": "a", "regex": "x", "alphabet": 'x"', "max_length": 1}, "must not contain"),
        ({"name": "a", "regex": "ACCT[0-9]+", "alphabet": "AC0123456789", "max_length": 20}, "'T'"),
        ({"name": "a", "regex": "x", "alphabet": "x", "max_length": 0}, "between 1 and"),
        (
            {"name": "a", "regex": "x", "alphabet": "x", "max_length": 1, "validator": "md5"},
            "one of",
        ),
        (
            {"name": "a", "regex": "x", "alphabet": "x", "start_chars": "y", "max_length": 1},
            "start_chars must be part",
        ),
    ],
)
def test_declared_detector_validation(spec: dict[str, object], match: str) -> None:
    with pytest.raises(PatternConfigurationError, match=match):
        build_pattern(spec)


def test_unverifiable_regex_is_accepted_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        pattern = build_pattern(
            {
                "name": "anything",
                "regex": "ID.[0-9]{4}",
                "alphabet": "ID0123456789-",
                "max_length": 7,
            }
        )
    assert pattern.name == "anything"
    assert any(r.event == "patterns.alphabet_unverified" for r in caplog.records)  # type: ignore[attr-defined]
    assert regex_alphabet(pattern.regex) is None


def test_pattern_documents_reject_duplicates_and_bad_shapes() -> None:
    spec = {"name": "dup", "regex": "x", "alphabet": "x", "max_length": 1}
    with pytest.raises(PatternConfigurationError, match="declared twice"):
        load_pattern_document({"patterns": [spec, spec]})
    with pytest.raises(PatternConfigurationError, match="non-empty"):
        load_pattern_document({"patterns": []})
    with pytest.raises(PatternConfigurationError, match="must be an object"):
        load_pattern_document({"patterns": ["x"]})
