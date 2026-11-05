"""PII detectors.

Each detector is a regular expression plus a *resolver*: a semantic check
(Luhn for card numbers, allocation rules for SSNs, length rules for e-mail
addresses) that turns a regex match into the exact span to redact, or
rejects it. A resolver may accept a shorter prefix of the match: a card
number followed by a CVV is matched greedily as 19 digits and resolves to the
16 digit card in front of the separator.

The detector also declares the character set a *partial* match can consist
of and the characters a match can start with. The streaming redactor uses
that metadata to decide how much of the tail of a chunk must be held back
until the next chunk arrives, which is what makes redaction correct across
chunk boundaries without buffering the response. ``max_length`` must be a
true upper bound on the length of an *accepted* span: it caps that holdback.

All detectors work on ASCII digits and letters on purpose: a Unicode-aware
``\\d`` would make the regex engine's idea of a digit differ from the
resolver's, and the streaming guarantees depend on both agreeing.
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

_ALNUM: Final = string.ascii_letters + string.digits
NBSP: Final = "\u00a0"
GROUP_SEPARATORS: Final = " \t\n\r-./_*`"
"""Characters that may separate the digit groups of a card number in the detection view
(every Unicode space is already a plain space there, every dash a hyphen): whitespace,
hyphen, dot, slash, underscore and markdown emphasis markers."""
_CARD_SEPARATOR_CLASS: Final = r"[ \t\n\r\-./_*`]"
_SSN_SEPARATORS: Final = " \t-./_"
_SSN_SEPARATOR_CLASS: Final = r"[ \t\-./_]"
MAX_CARD_DIGITS: Final = 19
MIN_CARD_DIGITS: Final = 13
MAX_CARD_GROUPS: Final = 5


@dataclass(frozen=True, slots=True)
class PiiPattern:
    name: str
    regex: re.Pattern[str]
    holdback_chars: frozenset[str]
    """Every character a complete or partial match may contain."""
    start_chars: frozenset[str]
    """Characters a match can begin with (used to trim separators from the holdback)."""
    max_length: int
    """Upper bound on the length of an accepted span; bounds the holdback and the scan window."""
    resolver: Callable[[str], list[tuple[int, int]]] | None = None
    """Maps a regex match to the ``(start, end)`` windows to redact inside it (none: not PII).

    A window may be shorter than the match: a card number followed by its CVV
    arrives as one digit run and resolves to the card in front of the separator;
    a run holding several cards resolves to all of them in one pass."""
    rejects_whole_match: bool = False
    """When the resolver finds nothing, nothing starting anywhere inside the match can be
    PII either (the resolver already tried every aligned window), so scanning may skip
    past the match instead of retrying one character later. Keeps long digit runs linear."""

    def resolve(self, text: str) -> list[tuple[int, int]]:
        if self.resolver is None:
            return [(0, len(text))]
        return self.resolver(text)

    def accepts(self, text: str) -> bool:
        """Whether the whole regex match is PII (convenience for tests and tooling)."""
        return self.resolve(text) == [(0, len(text))]


# --- email ------------------------------------------------------------------

EMAIL_MAX_LENGTH: Final = 254
_EMAIL_LOCAL: Final = r"[A-Za-z0-9._%+-]{1,64}"
_EMAIL_LABEL: Final = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"


def _email_span(text: str) -> list[tuple[int, int]]:
    # RFC 5321 bounds: 64 character local part (already in the regex) and 254 overall.
    return [(0, len(text))] if len(text) <= EMAIL_MAX_LENGTH else []


EMAIL = PiiPattern(
    name="email",
    regex=re.compile(rf"{_EMAIL_LOCAL}@(?:{_EMAIL_LABEL}\.)+[A-Za-z]{{2,63}}"),
    holdback_chars=frozenset(_ALNUM + "._%+-@"),
    start_chars=frozenset(_ALNUM + "._%+-"),
    max_length=EMAIL_MAX_LENGTH,
    resolver=_email_span,
)


# --- payment card numbers ---------------------------------------------------


def luhn_checksum_ok(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _is_card_number(digits: str) -> bool:
    return (
        MIN_CARD_DIGITS <= len(digits) <= MAX_CARD_DIGITS
        and len(set(digits)) > 1
        and (luhn_checksum_ok(digits))
    )


def _card_windows(text: str) -> list[tuple[int, int]]:
    """Every group-aligned window of ``text`` that is a valid card number, left to right.

    ``text`` is a run of 13 or more digits with separators. A window holds 13
    to 19 digits, starts at the run start or right after a separator and ends
    at the run end or right before one, so "4111 1111 1111 1111 123" (card
    plus CVV), "1 4111 ..." and "4111-1111-1111-1111-123" all resolve to the
    card itself, and a run holding several cards yields all of them. For each
    start the longest valid window wins; windows never overlap. Luhn sums come
    from two prefix-sum tables (one per digit parity), so one pass over the run
    is linear whatever its length.
    """
    positions = [i for i, ch in enumerate(text) if ch in string.digits]
    count = len(positions)
    if count < MIN_CARD_DIGITS:
        return []
    values = [ord(text[i]) - 48 for i in positions]
    plain = [[0], [0]]
    doubled = [[0], [0]]
    for index, value in enumerate(values):
        twice = value * 2
        twice = twice - 9 if twice > 9 else twice
        for parity in (0, 1):
            plain[parity].append(plain[parity][-1] + (value if index % 2 == parity else 0))
            doubled[parity].append(doubled[parity][-1] + (twice if index % 2 == parity else 0))
    # same_until[k]: first digit index after k whose value differs (windows made of one
    # repeated digit are never cards, whatever their checksum).
    same_until = [count] * count
    for k in range(count - 2, -1, -1):
        same_until[k] = k + 1 if values[k + 1] != values[k] else same_until[k + 1]
    starts = [k for k, i in enumerate(positions) if i == 0 or text[i - 1] not in string.digits]
    ends = [
        k + 1
        for k, i in enumerate(positions)
        if i + 1 == len(text) or text[i + 1] not in string.digits
    ]
    windows: list[tuple[int, int]] = []
    taken_until = 0  # digit index up to which an accepted window reaches
    end_cursor = 0
    for start in starts:
        if start < taken_until:
            continue
        while end_cursor < len(ends) and ends[end_cursor] <= start:
            end_cursor += 1
        # Real card formats have at most five digit groups (4-4-4-4-3 is the longest),
        # so only the next few group boundaries are candidate ends. That keeps lists of
        # small numbers ("12 15 18 20 ...") from being carved into pseudo cards and keeps
        # the work per start constant.
        candidates = [
            end
            for end in ends[end_cursor : end_cursor + MAX_CARD_GROUPS]
            if MIN_CARD_DIGITS <= end - start <= MAX_CARD_DIGITS
        ]
        for end in reversed(candidates):
            if same_until[start] >= end:
                continue
            # Digits at index j with j and end of equal parity are doubled.
            parity = end % 2
            total = (doubled[parity][end] - doubled[parity][start]) + (
                plain[1 - parity][end] - plain[1 - parity][start]
            )
            if total % 10 == 0:
                windows.append((positions[start], positions[end - 1] + 1))
                taken_until = end
                break
    return windows


CREDIT_CARD = PiiPattern(
    name="credit_card",
    # The whole separated digit run is one candidate; the resolver finds the cards in it.
    regex=re.compile(
        rf"(?<![0-9])(?:[0-9]{_CARD_SEPARATOR_CLASS}{{0,3}}){{{MIN_CARD_DIGITS - 1},}}"
        rf"[0-9](?!{_CARD_SEPARATOR_CLASS}{{0,3}}[0-9])"
    ),
    holdback_chars=frozenset(string.digits + GROUP_SEPARATORS),
    start_chars=frozenset(string.digits),
    # Bounds the holdback, not the match: a window is at most 19 digits with up to
    # three separator characters between each pair, so any window still growing
    # lies within the last 73 characters of a run.
    max_length=MAX_CARD_DIGITS + (MAX_CARD_DIGITS - 1) * 3,
    resolver=_card_windows,
    rejects_whole_match=True,
)


# --- US social security numbers --------------------------------------------


def _ssn_allocation_ok(area: str, group: str, serial: str) -> bool:
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def _ssn_span(text: str) -> list[tuple[int, int]]:
    digits = "".join(ch for ch in text if ch in string.digits)
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    return [(0, len(text))] if _ssn_allocation_ok(area, group, serial) else []


SSN = PiiPattern(
    name="ssn",
    regex=re.compile(
        rf"(?<![0-9])[0-9]{{3}}{_SSN_SEPARATOR_CLASS}{{1,2}}[0-9]{{2}}"
        rf"{_SSN_SEPARATOR_CLASS}{{1,2}}[0-9]{{4}}(?![0-9])"
    ),
    holdback_chars=frozenset(string.digits + _SSN_SEPARATORS),
    start_chars=frozenset(string.digits),
    max_length=13,
    resolver=_ssn_span,
)


DEFAULT_PATTERNS: Final[tuple[PiiPattern, ...]] = (EMAIL, CREDIT_CARD, SSN)
PATTERNS_BY_NAME: Final[dict[str, PiiPattern]] = {p.name: p for p in DEFAULT_PATTERNS}


# --- optional presets ---------------------------------------------------------
#
# Not enabled by default: each trades some precision for recall and belongs in
# deployments whose traffic warrants it (``LLM_GATEWAY_PATTERNS``).

_IBAN_LETTERS: Final = string.ascii_uppercase
_IBAN_MIN_DIGITS: Final = 15
_IBAN_MAX_DIGITS: Final = 34


def iban_checksum_ok(compact: str) -> bool:
    """ISO 13616 mod 97 check on an IBAN without separators."""
    if not (_IBAN_MIN_DIGITS <= len(compact) <= _IBAN_MAX_DIGITS):
        return False
    if not (compact[:2].isalpha() and compact[2:4].isdigit()):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in rearranged)
    return int(numeric) % 97 == 1


def _iban_span(text: str) -> list[tuple[int, int]]:
    compact = text.replace(" ", "").replace(NBSP, "")
    return [(0, len(text))] if iban_checksum_ok(compact) else []


IBAN = PiiPattern(
    name="iban",
    regex=re.compile(
        rf"(?<![A-Z0-9])[A-Z]{{2}}[0-9]{{2}}(?:[ {NBSP}]?[A-Z0-9]){{11,30}}(?![A-Z0-9])"
    ),
    holdback_chars=frozenset(_IBAN_LETTERS + string.digits + " " + NBSP),
    start_chars=frozenset(_IBAN_LETTERS),
    max_length=42,  # 34 characters plus up to 8 group separators
    resolver=_iban_span,
)


def _phone_us_span(text: str) -> list[tuple[int, int]]:
    digits = "".join(ch for ch in text if ch in string.digits)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10 or digits[0] in "01" or digits[3] in "01":
        return []
    if digits[1:3] == "11":  # N11 service codes are not area codes
        return []
    return [(0, len(text))]


PHONE_US = PiiPattern(
    name="phone_us",
    regex=re.compile(
        rf"(?<![0-9])(?:\+?1[ .{NBSP}-]?)?\(?[2-9][0-9]{{2}}\)?[ .{NBSP}-]?[2-9][0-9]{{2}}"
        rf"[ .{NBSP}-]?[0-9]{{4}}(?![0-9])"
    ),
    holdback_chars=frozenset(string.digits + " .()+-" + NBSP),
    start_chars=frozenset(string.digits + "+("),
    max_length=20,
    resolver=_phone_us_span,
)

PHONE_E164 = PiiPattern(
    name="phone_e164",
    regex=re.compile(r"(?<![0-9+])\+[1-9][0-9]{7,14}(?![0-9])"),
    holdback_chars=frozenset(string.digits + "+"),
    start_chars=frozenset("+"),
    max_length=16,
)


def _ssn_compact_span(text: str) -> list[tuple[int, int]]:
    return [(0, len(text))] if _ssn_allocation_ok(text[:3], text[3:5], text[5:]) else []


SSN_COMPACT = PiiPattern(
    name="ssn_compact",
    regex=re.compile(r"(?<![0-9-])[0-9]{9}(?![0-9-])"),
    holdback_chars=frozenset(string.digits),
    start_chars=frozenset(string.digits),
    max_length=9,
    resolver=_ssn_compact_span,
)

PRESET_PATTERNS: Final[tuple[PiiPattern, ...]] = (IBAN, PHONE_US, PHONE_E164, SSN_COMPACT)
DEFAULT_PATTERN_NAMES: Final[tuple[str, ...]] = tuple(p.name for p in DEFAULT_PATTERNS)
PATTERNS_BY_NAME.update({p.name: p for p in PRESET_PATTERNS})
