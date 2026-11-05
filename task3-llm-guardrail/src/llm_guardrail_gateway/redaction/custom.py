"""Deployment specific detectors, declared in a JSON file.

A fine-tuned model reproduces the identifiers of the data it was trained on:
customer numbers, account numbers, internal ticket ids. Those shapes are not
knowable in advance, so operators declare them::

    {"patterns": [
      {"name": "customer_id", "regex": "CUST-[0-9]{5}", "alphabet": "CUST-0123456789",
       "start_chars": "C", "max_length": 10},
      {"name": "account", "regex": "(?<![0-9])[0-9]{10,12}(?![0-9])", "alphabet": "0123456789",
       "max_length": 12, "validator": "luhn"}
    ]}

The streaming guarantee depends on ``alphabet`` really containing every
character a match can consist of. The declared alphabet is therefore checked
against the compiled regex: when the regex's character set can be determined
statically (literals, classes, ranges, ASCII categories) a mismatch is a
configuration error; constructs whose set cannot be bounded (``.``, negated
classes, back references, Unicode categories) are accepted with a warning,
because then the declaration is the only source of truth.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import string
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Final

from llm_guardrail_gateway.redaction.patterns import (
    PATTERNS_BY_NAME,
    PiiPattern,
    iban_checksum_ok,
    luhn_checksum_ok,
)

logger = logging.getLogger(__name__)

# The standard library's own regex parser: the only way to inspect a pattern's
# structure. Private, but it is what ``re.compile`` itself runs, and the former
# public aliases (``sre_parse``) now only emit deprecation warnings around it.
sre_parser: Any = importlib.import_module("re._parser")
sre_constants: Any = importlib.import_module("re._constants")

_NAME_RE: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FORBIDDEN_ALPHABET_CHARS: Final = frozenset('"\\{}[]\n\r')
"""Characters a span may never contain: they would break the JSON context tracking."""
_MAX_SPAN_LENGTH: Final = 1024
_ASCII_DIGITS: Final = frozenset(string.digits)
_ASCII_WORD: Final = frozenset(string.ascii_letters + string.digits + "_")
_ASCII_SPACE: Final = frozenset(" \t\n\r\f\v")
_MAX_RANGE: Final = 512


class PatternConfigurationError(ValueError):
    """A declared detector is unusable."""


def _luhn_digits(text: str) -> list[tuple[int, int]]:
    digits = "".join(ch for ch in text if ch in string.digits)
    return [(0, len(text))] if len(digits) >= 2 and luhn_checksum_ok(digits) else []


def _mod97(text: str) -> list[tuple[int, int]]:
    compact = "".join(ch for ch in text if ch.isalnum())
    return [(0, len(text))] if iban_checksum_ok(compact) else []


VALIDATORS: Final[dict[str, Callable[[str], list[tuple[int, int]]]]] = {
    "luhn": _luhn_digits,
    "mod97": _mod97,
}


def regex_alphabet(pattern: re.Pattern[str]) -> frozenset[str] | None:
    """Every character the pattern can consume, or ``None`` when that set is unbounded."""
    try:
        tree = sre_parser.parse(pattern.pattern, pattern.flags)
    except re.error:  # pragma: no cover - the pattern compiled already
        return None
    ignore_case = bool(pattern.flags & re.IGNORECASE)
    ascii_only = bool(pattern.flags & re.ASCII)
    chars: set[str] = set()

    def add(char: str) -> None:
        chars.add(char)
        if ignore_case:
            chars.add(char.lower())
            chars.add(char.upper())

    def walk(items: Iterable[Any]) -> bool:
        for opcode, argument in items:
            if opcode is sre_constants.LITERAL:
                add(chr(argument))
            elif opcode is sre_constants.IN:
                if not walk_set(argument):
                    return False
            elif opcode in (sre_constants.ANY, sre_constants.NOT_LITERAL):
                return False
            elif opcode is sre_constants.BRANCH:
                for alternative in argument[1]:
                    if not walk(alternative):
                        return False
            elif opcode is sre_constants.SUBPATTERN:
                if not walk(argument[3]):
                    return False
            elif opcode in (
                sre_constants.MAX_REPEAT,
                sre_constants.MIN_REPEAT,
                sre_constants.POSSESSIVE_REPEAT,
            ):
                if not walk(argument[2]):
                    return False
            elif opcode is sre_constants.ATOMIC_GROUP:
                if not walk(argument):
                    return False
            elif opcode in (sre_constants.AT, sre_constants.ASSERT, sre_constants.ASSERT_NOT):
                continue  # anchors and look-arounds consume nothing
            else:
                return False  # back references and anything exotic
        return True

    def walk_set(items: Iterable[Any]) -> bool:
        for opcode, argument in items:
            if opcode is sre_constants.LITERAL:
                add(chr(argument))
            elif opcode is sre_constants.RANGE:
                low, high = argument
                if high - low > _MAX_RANGE:
                    return False
                for code in range(low, high + 1):
                    add(chr(code))
            elif opcode is sre_constants.CATEGORY:
                if not ascii_only:
                    return False
                if argument is sre_constants.CATEGORY_DIGIT:
                    chars.update(_ASCII_DIGITS)
                elif argument is sre_constants.CATEGORY_WORD:
                    chars.update(_ASCII_WORD)
                elif argument is sre_constants.CATEGORY_SPACE:
                    chars.update(_ASCII_SPACE)
                else:
                    return False  # negated categories
            else:
                return False  # NEGATE and friends
        return True

    return frozenset(chars) if walk(tree) else None


def build_pattern(spec: dict[str, Any]) -> PiiPattern:
    """Turn one JSON object into a validated ``PiiPattern``."""
    name = spec.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise PatternConfigurationError("pattern name must match [a-z][a-z0-9_]* (max 64 chars)")
    if name in PATTERNS_BY_NAME:
        raise PatternConfigurationError(f"pattern {name!r} shadows a built-in detector")
    regex_text = spec.get("regex")
    if not isinstance(regex_text, str) or not regex_text:
        raise PatternConfigurationError(f"pattern {name!r}: regex is required")
    flags = re.IGNORECASE if spec.get("case_insensitive") else 0
    try:
        regex = re.compile(regex_text, flags)
    except re.error as exc:
        raise PatternConfigurationError(f"pattern {name!r}: invalid regex: {exc}") from exc
    alphabet_text = spec.get("alphabet")
    if not isinstance(alphabet_text, str) or not alphabet_text:
        raise PatternConfigurationError(f"pattern {name!r}: alphabet is required")
    alphabet = frozenset(alphabet_text)
    if flags & re.IGNORECASE:
        alphabet = frozenset(
            alphabet | {c.lower() for c in alphabet} | {c.upper() for c in alphabet}
        )
    forbidden = alphabet & _FORBIDDEN_ALPHABET_CHARS
    if forbidden:
        raise PatternConfigurationError(
            f"pattern {name!r}: alphabet must not contain {sorted(forbidden)!r}"
        )
    start_text = spec.get("start_chars", alphabet_text)
    if not isinstance(start_text, str) or not start_text:
        raise PatternConfigurationError(f"pattern {name!r}: start_chars must be a non-empty string")
    start_chars = frozenset(start_text)
    if flags & re.IGNORECASE:
        start_chars = frozenset(
            start_chars | {c.lower() for c in start_chars} | {c.upper() for c in start_chars}
        )
    if not start_chars <= alphabet:
        raise PatternConfigurationError(f"pattern {name!r}: start_chars must be part of alphabet")
    max_length = spec.get("max_length")
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise PatternConfigurationError(f"pattern {name!r}: max_length must be an integer")
    if not 1 <= max_length <= _MAX_SPAN_LENGTH:
        raise PatternConfigurationError(
            f"pattern {name!r}: max_length must be between 1 and {_MAX_SPAN_LENGTH}"
        )
    validator_name = spec.get("validator")
    resolver: Callable[[str], list[tuple[int, int]]] | None = None
    if validator_name is not None:
        if validator_name not in VALIDATORS:
            raise PatternConfigurationError(
                f"pattern {name!r}: validator must be one of {sorted(VALIDATORS)}"
            )
        resolver = VALIDATORS[validator_name]

    consumed = regex_alphabet(regex)
    if consumed is None:
        logger.warning(
            "pattern alphabet could not be verified statically",
            extra={"event": "patterns.alphabet_unverified", "pattern": name},
        )
    elif not consumed <= alphabet:
        missing = "".join(sorted(consumed - alphabet))
        raise PatternConfigurationError(
            f"pattern {name!r}: regex can match characters outside its alphabet: {missing!r}"
        )
    return PiiPattern(
        name=name,
        regex=regex,
        holdback_chars=alphabet,
        start_chars=start_chars,
        max_length=max_length,
        resolver=resolver,
    )


def load_pattern_document(document: Any) -> tuple[PiiPattern, ...]:
    entries = document.get("patterns") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise PatternConfigurationError(
            'pattern document must be an object with a non-empty "patterns" array'
        )
    patterns: list[PiiPattern] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PatternConfigurationError("every pattern entry must be an object")
        pattern = build_pattern(entry)
        if pattern.name in seen:
            raise PatternConfigurationError(f"pattern {pattern.name!r} is declared twice")
        seen.add(pattern.name)
        patterns.append(pattern)
    return tuple(patterns)


def load_pattern_file(path: Path) -> tuple[PiiPattern, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PatternConfigurationError(f"cannot read pattern file {path}: {exc}") from exc
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise PatternConfigurationError(f"pattern file {path} is not valid JSON: {exc}") from exc
    return load_pattern_document(document)
