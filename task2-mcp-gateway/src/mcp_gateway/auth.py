"""Bearer token authentication.

The gateway terminates authentication: it maps a bearer token to a
``Principal`` (subject and role) and never forwards the caller's credential
to the downstream server. ``TokenVerifier`` is a small protocol so the static
token map used here can be swapped for a JWT or introspection based verifier
without touching the request pipeline.
"""

from __future__ import annotations

import hmac
import json
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

# RFC 6750 section 2.1: "Bearer" is case-insensitive, followed by one or more
# spaces and a token68 value.
_BEARER_RE = re.compile(r"^Bearer[ ]+([A-Za-z0-9\-._~+/]+=*)$", re.IGNORECASE)
_TOKEN68_RE: Final = re.compile(r"[A-Za-z0-9\-._~+/]+=*")
_IDENTITY_RE: Final = re.compile(r"[\x21-\x7e](?:[\x20-\x7e]{0,126}[\x21-\x7e])?")
"""Subjects and roles travel in request headers: printable ASCII, 1 to 128 characters,
no leading or trailing spaces."""
MIN_TOKEN_LENGTH: Final = 32
"""Tokens are bearer secrets; ``secrets.token_urlsafe(32)`` is the suggested generator."""
DEVELOPMENT_MARKER: Final = "development_only"
"""Top-level flag in a token document whose tokens are published (the repository's
development file); such a document is refused unless explicitly allowed."""
_FINGERPRINT_KEY: Final = secrets.token_bytes(32)


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    role: str
    token_fingerprint: str
    """Short keyed hash of the token, safe to log for correlation within one process."""


class AuthenticationError(Exception):
    """The request could not be attributed to a principal."""

    def __init__(self, reason: str, description: str) -> None:
        super().__init__(description)
        self.reason = reason
        self.description = description

    @property
    def www_authenticate(self) -> str:
        if self.reason == "missing_token":
            return 'Bearer realm="mcp-gateway"'
        return (
            'Bearer realm="mcp-gateway", error="invalid_token", '
            f'error_description="{self.description}"'
        )


class TokenVerifier(Protocol):
    def verify(self, token: str) -> Principal | None: ...


def fingerprint(token: str) -> str:
    """A log-safe correlation id for a token.

    Keyed with a random per-process secret, so a fingerprint read from a log
    cannot be used offline to confirm a guessed token. The price is that
    fingerprints only correlate within one process lifetime.
    """
    return hmac.new(_FINGERPRINT_KEY, token.encode("utf-8"), "sha256").hexdigest()[:12]


def extract_bearer_token(authorization: str | None) -> str:
    """Return the token from an ``Authorization`` header or raise ``AuthenticationError``."""
    if authorization is None or not authorization.strip():
        raise AuthenticationError(
            "missing_token", "Authorization header with a Bearer token is required"
        )
    match = _BEARER_RE.match(authorization.strip())
    if match is None:
        raise AuthenticationError(
            "malformed_header", "Authorization header must be 'Bearer <token>'"
        )
    return match.group(1)


def authenticate(authorization: str | None, verifier: TokenVerifier) -> Principal:
    token = extract_bearer_token(authorization)
    principal = verifier.verify(token)
    if principal is None:
        raise AuthenticationError("invalid_token", "The bearer token is not recognised")
    return principal


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse documents that list one token twice; which role wins must never depend
    on file order. The offending key is named by position, never quoted."""
    document: dict[str, Any] = {}
    for position, (key, value) in enumerate(pairs, start=1):
        if key in document:
            raise ValueError(f"duplicate member: entry {position} of an object repeats a name")
        document[key] = value
    return document


class StaticTokenVerifier:
    """Verifier backed by an in-memory token map (loaded from JSON configuration).

    Lookups compare the presented token against every configured token with a
    constant-time comparison so response timing does not reveal which prefixes
    exist. Validation messages never quote token material.
    """

    def __init__(self, tokens: Mapping[str, Mapping[str, Any]]) -> None:
        if not tokens:
            raise ValueError("token map must contain at least one token")
        entries: dict[bytes, tuple[str, str]] = {}
        for position, (token, identity) in enumerate(tokens.items(), start=1):
            if (
                not isinstance(token, str)
                or len(token) < MIN_TOKEN_LENGTH
                or _TOKEN68_RE.fullmatch(token) is None
            ):
                raise ValueError(
                    f"token entry {position}: tokens must be at least {MIN_TOKEN_LENGTH} "
                    "characters from the RFC 6750 token68 alphabet "
                    "(letters, digits, - . _ ~ + / and trailing =)"
                )
            if not isinstance(identity, Mapping):
                raise ValueError(f"token entry {position}: identity must be an object")
            subject = identity.get("subject")
            role = identity.get("role")
            if not isinstance(subject, str) or not _IDENTITY_RE.fullmatch(subject):
                raise ValueError(
                    f"token entry {position}: lacks a subject (1 to 128 printable ASCII characters)"
                )
            if not isinstance(role, str) or not _IDENTITY_RE.fullmatch(role):
                raise ValueError(
                    f"token for subject {subject!r} lacks a role "
                    "(1 to 128 printable ASCII characters)"
                )
            entries[token.encode("utf-8")] = (subject, role)
        self._entries = entries

    @classmethod
    def from_json(cls, text: str, *, allow_development: bool = False) -> StaticTokenVerifier:
        document = json.loads(text, object_pairs_hook=_pairs_without_duplicates)
        tokens = document.get("tokens") if isinstance(document, dict) else None
        if not isinstance(tokens, dict):
            raise ValueError('token document must be an object with a "tokens" mapping')
        if document.get(DEVELOPMENT_MARKER) is True and not allow_development:
            raise ValueError(
                f"the token document is marked {DEVELOPMENT_MARKER}: its tokens are published; "
                "set MCP_GATEWAY_ALLOW_DEVELOPMENT_CREDENTIALS=true to use it outside production"
            )
        return cls(tokens)

    @classmethod
    def from_json_file(cls, path: Path, *, allow_development: bool = False) -> StaticTokenVerifier:
        return cls.from_json(path.read_text(encoding="utf-8"), allow_development=allow_development)

    def verify(self, token: str) -> Principal | None:
        presented = token.encode("utf-8")
        matched: tuple[str, str] | None = None
        for known, identity in self._entries.items():
            # No early exit: every candidate is compared.
            if hmac.compare_digest(known, presented):
                matched = identity
        if matched is None:
            return None
        subject, role = matched
        return Principal(subject=subject, role=role, token_fingerprint=fingerprint(token))
