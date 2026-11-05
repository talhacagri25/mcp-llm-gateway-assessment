"""The detection view: a folded copy of the text that detectors scan.

A guardrail that matches the literal characters is defeated by representation
tricks a model can be coaxed into: zero-width characters between the digits of
a card number, fullwidth or Arabic-Indic digits, en dashes or figure spaces as
separators, combining marks glued to a digit, control characters inside a
token, Cyrillic look-alikes in a domain. Detectors therefore never see the raw
text. They see a folded view in which

* format characters (``Cf``: zero-width space and joiner, bidi marks, soft
  hyphen, word joiner) and combining marks (``Mn``, ``Me``, ``Mc``) are dropped,
* control characters other than tab, newline, carriage return, vertical tab
  and form feed are dropped,
* every Unicode decimal digit becomes its ASCII digit,
* every space separator (``Zs``) becomes a plain space and every dash
  (``Pd``, plus the minus sign) a plain hyphen,
* the common Cyrillic and Greek look-alikes of Latin letters become those
  letters (a skeleton in the sense of Unicode TR39, restricted to the pairs
  that render identically in ordinary fonts),
* everything else is folded with NFKC, which maps fullwidth and other
  compatibility forms to ASCII where one exists.

Each folded character remembers the index of the original character it came
from, so a span found in the folded view maps back to the exact original
range, dropped characters included, and the replacement removes the whole
disguise. ASCII text without control characters is its own view, so the
common case costs nothing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

_DROP_CATEGORIES: Final = frozenset({"Cf", "Mn", "Me", "Mc"})
_KEPT_CONTROLS: Final = frozenset("\t\n\r\x0b\x0c")
_ASCII_CONTROL: Final = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_CONFUSABLES: Final[dict[str, str]] = {
    # Cyrillic and Greek letters that render like Latin ones, as code point escapes
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0445": "x",
    "\u0456": "i",
    "\u0458": "j",
    "\u0455": "s",
    "\u04bb": "h",
    "\u0501": "d",
    "\u051b": "q",
    "\u051d": "w",
    "\u04cf": "l",
    "\u043a": "k",
    "\u0442": "t",
    "\u043c": "m",
    "\u043d": "h",
    "\u0432": "b",
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0423": "Y",
    "\u0425": "X",
    "\u0406": "I",
    "\u0408": "J",
    "\u0405": "S",
    "\u03b1": "a",
    "\u03bf": "o",
    "\u03bd": "v",
    "\u03c1": "p",
    "\u03c4": "t",
    "\u03c5": "u",
    "\u03b9": "i",
    "\u03ba": "k",
    "\u03c7": "x",
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0396": "Z",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a5": "Y",
    "\u03a7": "X",
}
_cache: dict[str, str] = {}

Origin = list[int] | None
"""Original index of every folded character; ``None`` means the view is the text itself."""


def fold_char(char: str) -> str:
    """Folded form of one character (possibly empty, possibly several characters)."""
    cached = _cache.get(char)
    if cached is not None:
        return cached
    category = unicodedata.category(char)
    if category in _DROP_CATEGORIES or (category == "Cc" and char not in _KEPT_CONTROLS):
        folded = ""
    elif category == "Zs":
        folded = " "
    elif category == "Pd" or char == "\u2212":
        folded = "-"
    elif category == "Nd":
        folded = str(unicodedata.digit(char))
    elif char in _CONFUSABLES:
        folded = _CONFUSABLES[char]
    else:
        folded = unicodedata.normalize("NFKC", char)
        if folded != char:
            # Compatibility forms may expand (ligatures) or land on characters that
            # themselves need folding (fullwidth digits are Nd after NFKC).
            folded = "".join(c if c.isascii() else fold_char(c) for c in folded)
    if len(_cache) < 65536:
        _cache[char] = folded
    return folded


def fold(text: str) -> tuple[str, Origin]:
    """The detection view of ``text`` and the origin index of each of its characters."""
    if text.isascii() and _ASCII_CONTROL.search(text) is None:
        return text, None
    pieces: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(text):
        folded = char if char.isascii() and not _ASCII_CONTROL.match(char) else fold_char(char)
        for out in folded:
            pieces.append(out)
            origin.append(index)
    return "".join(pieces), origin


def to_original(origin: Origin, index: int, folded_length: int, original_length: int) -> int:
    """Original index of folded position ``index``; the end of the text maps to its end."""
    if index >= folded_length:
        return original_length
    return index if origin is None else origin[index]
