"""Chunk-boundary safe redaction of a text stream.

The problem: a token stream delivers ``"contact jo"``, ``"hn@exa"``,
``"mple.com now"``. Scanning each chunk on its own finds nothing. Buffering
the whole response finds the address but destroys streaming.

The approach: after appending a chunk, release everything up to a *safe
boundary* and keep only the tail. The boundary is chosen so that

1. no match that could still grow with more input is released: for every
   detector, the trailing run of characters drawn from its alphabet (trimmed
   to begin with a character a match can start with, and capped by the
   detector's maximum span length) stays pending, because any match that
   extends past the end of the buffer consists solely of such characters;
2. no match that is already visible in the buffer is cut in two: if a match
   straddles the boundary, the boundary moves back to its start.

Everything before the boundary is redacted and emitted immediately; the tail
waits for the next chunk (or ``flush``).

Scanning is alignment independent by construction: every start position is
tried (a rejected candidate never swallows the text after it), each detector
resolves a candidate deterministically from its start, and one character of
released context is kept so look-behind assertions see the same neighbour the
one-shot scan would. That is what makes the result identical for every
chunking.

Detectors scan the *detection view* of the text (``folding.py``): format
characters and combining marks removed, Unicode digits, spaces and dashes
folded to ASCII, compatibility forms normalised. Spans found there are
mapped back to the original text, so a card number written with zero-width
spaces between its digits or in fullwidth digits is redacted whole,
disguise included.

Properties:

* **Chunking invariance**: the concatenated outputs of ``feed`` calls plus
  ``flush`` equal a one-shot redaction of the whole text, for any chunking.
* **Bounded memory**: the tail never exceeds ``max(max_holdback, L) + L``
  characters, where ``L`` is the longest span any detector can accept.
* **Low latency**: text up to and including the last whitespace is released in
  the call it arrived in; at most one trailing word or number group waits for
  the next chunk.

JSON awareness: a match that is a bare JSON value (a card number emitted as a
number rather than a string) is replaced by a JSON string so the document
stays parseable. Channels that carry a JSON document (tool call arguments)
are created with ``json_mode=True``; prose channels detect embedded JSON
objects on their own (a tool call written into the text, a fenced JSON block),
tracking braces and string literals from the first ``{`` to its closing
``}``. Text outside any object is never touched beyond the replacement itself.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Final

from llm_guardrail_gateway.redaction.folding import Origin, fold, to_original
from llm_guardrail_gateway.redaction.patterns import DEFAULT_PATTERNS, PiiPattern

DEFAULT_REPLACEMENT: Final = "[REDACTED]"
DEFAULT_MAX_HOLDBACK: Final = 256

Span = tuple[int, int, str]


class _JsonContext:
    """Tracks whether released text currently sits inside JSON, and inside a string literal.

    With ``whole_document`` the whole stream is JSON (tool arguments). Otherwise
    JSON context begins at an opening brace and ends when the object closes;
    prose in between is ignored, so quotes in ordinary text never confuse the
    tracker. Only the characters that have been released are fed, in order.
    """

    __slots__ = ("_depth", "_escaped", "_in_string", "_last_significant", "_whole_document")

    def __init__(self, whole_document: bool) -> None:
        self._whole_document = whole_document
        self._depth = 0
        self._in_string = False
        self._escaped = False
        self._last_significant = ""

    @property
    def in_json(self) -> bool:
        return self._whole_document or self._depth > 0

    def advance(self, text: str) -> None:
        for char in text:
            if not self.in_json:
                if char == "{":
                    self._depth = 1
                    self._in_string = False
                    self._escaped = False
                    self._last_significant = char
                continue
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
                continue
            if char == '"':
                self._in_string = True
            elif char in "{[":
                self._depth += 1
            elif char in "}]":
                self._depth = max(0, self._depth - 1)
            if not char.isspace():
                self._last_significant = char

    def bare_value_here(self) -> bool:
        """Would a value starting now be a bare JSON value (not inside a string literal)?"""
        if not self.in_json or self._in_string:
            return False
        if self._last_significant in (":", "[", ","):
            return True
        return self._whole_document and self._last_significant == ""


class StreamingRedactor:
    def __init__(
        self,
        patterns: Sequence[PiiPattern] = DEFAULT_PATTERNS,
        *,
        replacement: str = DEFAULT_REPLACEMENT,
        max_holdback: int = DEFAULT_MAX_HOLDBACK,
        json_mode: bool = False,
    ) -> None:
        if max_holdback < 1:
            raise ValueError("max_holdback must be at least 1")
        if not patterns:
            raise ValueError("at least one pattern is required")
        self._patterns = tuple(patterns)
        self._replacement = replacement
        self._quoted_replacement = json.dumps(replacement)
        # The cap can never be allowed to cut through a single match, so it is
        # at least as long as the longest span any detector can accept.
        self._max_holdback = max(max_holdback, max(p.max_length for p in self._patterns))
        self._json = _JsonContext(whole_document=json_mode)
        self._pending = ""
        self._context = ""
        """The last released character, in the detection view (look-behind context)."""
        self.redactions: Counter[str] = Counter()

    @property
    def pending(self) -> str:
        """Text held back awaiting more input (exposed for tests and metrics)."""
        return self._pending

    def feed(self, chunk: str) -> str:
        """Absorb ``chunk`` and return the text that is now safe to emit."""
        if not chunk:
            return ""
        buffer = self._pending + chunk
        view, origin = fold(buffer)
        scan = self._context + view
        offset = len(self._context)
        spans = self._find_spans(scan, offset)

        boundary = self._growth_boundary(view)
        for start, end, _name in spans:
            if start - offset < boundary < end - offset:
                boundary = start - offset  # never split a visible match
                break

        if boundary == 0:
            self._pending = buffer
            return ""
        cut = to_original(origin, boundary, len(view), len(buffer))
        released, self._pending = buffer[:cut], buffer[cut:]
        releasable = [
            self._original_span(span, offset, origin, len(view), len(buffer))
            for span in spans
            if span[1] - offset <= boundary
        ]
        output = self._apply(released, releasable)
        self._context = view[boundary - 1]
        return output

    def flush(self) -> str:
        """Release whatever is still held back (call at end of stream)."""
        if not self._pending:
            return ""
        pending, self._pending = self._pending, ""
        view, origin = fold(pending)
        scan = self._context + view
        offset = len(self._context)
        spans = [
            self._original_span(span, offset, origin, len(view), len(pending))
            for span in self._find_spans(scan, offset)
        ]
        output = self._apply(pending, spans)
        if view:
            self._context = view[-1]
        return output

    @staticmethod
    def _original_span(
        span: Span, offset: int, origin: Origin, view_length: int, original_length: int
    ) -> Span:
        """Map a span in scan coordinates back to the original buffer."""
        start, end, name = span
        first = to_original(origin, start - offset, view_length, original_length)
        last = to_original(origin, end - offset - 1, view_length, original_length)
        return first, last + 1, name

    # -- internals -----------------------------------------------------------

    def _growth_boundary(self, buffer: str) -> int:
        """Start of the suffix that could still be extended into a match."""
        boundary = len(buffer)
        for pattern in self._patterns:
            boundary = min(boundary, self._trailing_run_start(buffer, pattern))
            if boundary == 0:
                break
        if len(buffer) - boundary > self._max_holdback:
            boundary = len(buffer) - self._max_holdback
        return boundary

    @staticmethod
    def _trailing_run_start(buffer: str, pattern: PiiPattern) -> int:
        end = len(buffer)
        start = end
        floor = max(0, end - pattern.max_length)
        while start > floor and buffer[start - 1] in pattern.holdback_chars:
            start -= 1
        while start < end and buffer[start] not in pattern.start_chars:
            start += 1
        return start

    def _find_spans(self, scan: str, offset: int) -> list[Span]:
        """Resolved, non-overlapping spans starting at or after ``offset``, sorted by start.

        Every start position is tried: after a rejected candidate the search
        resumes one character later instead of after the candidate, so a
        rejected digit run cannot hide a card number inside it. Searching with a
        start position (rather than slicing) keeps look-behind assertions aware
        of the context character.
        """
        candidates: list[Span] = []
        for pattern in self._patterns:
            position = offset
            while position <= len(scan):
                match = pattern.regex.search(scan, position)
                if match is None:
                    break
                start = match.start()
                windows = pattern.resolve(match.group(0))
                if not windows:
                    position = match.end() if pattern.rejects_whole_match else start + 1
                    continue
                for window_start, window_end in windows:
                    candidates.append((start + window_start, start + window_end, pattern.name))
                last_end = start + windows[-1][1]
                position = max(match.end(), last_end) if pattern.rejects_whole_match else last_end
        candidates.sort(key=lambda span: (span[0], -span[1]))
        resolved: list[Span] = []
        last_end = -1
        for span in candidates:
            if span[0] >= last_end:
                resolved.append(span)
                last_end = span[1]
        return resolved

    def _apply(self, text: str, spans: list[Span]) -> str:
        """Return ``text`` with ``spans`` (in its own coordinates) replaced."""
        if not spans:
            self._json.advance(text)
            return text
        pieces: list[str] = []
        cursor = 0
        for span_start, span_end, name in spans:
            between = text[cursor:span_start]
            self._json.advance(between)
            pieces.append(between)
            if self._json.bare_value_here():
                # A bare JSON value: keep the document parseable.
                pieces.append(self._quoted_replacement)
            else:
                pieces.append(self._replacement)
            # Detector alphabets contain no quotes, braces or backslashes, so the
            # span itself never changes the JSON state; feeding it keeps the
            # "last significant character" honest for an adjacent match.
            self._json.advance(text[span_start:span_end])
            self.redactions[name] += 1
            cursor = span_end
        tail = text[cursor:]
        self._json.advance(tail)
        pieces.append(tail)
        return "".join(pieces)


def redact_text(
    text: str,
    patterns: Sequence[PiiPattern] = DEFAULT_PATTERNS,
    *,
    replacement: str = DEFAULT_REPLACEMENT,
    json_mode: bool = False,
) -> str:
    """One-shot redaction, used for non-streaming responses and as the test oracle."""
    redactor = StreamingRedactor(
        patterns, replacement=replacement, max_holdback=max(1, len(text)), json_mode=json_mode
    )
    return redactor.feed(text) + redactor.flush()
