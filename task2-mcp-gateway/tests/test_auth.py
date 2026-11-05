"""Bearer token parsing and verification."""

from __future__ import annotations

import json

import pytest

from mcp_gateway.auth import (
    AuthenticationError,
    StaticTokenVerifier,
    authenticate,
    extract_bearer_token,
    fingerprint,
)


@pytest.fixture
def verifier() -> StaticTokenVerifier:
    return StaticTokenVerifier(
        {
            "admin-token-0123456789abcdefghijklmnop": {"subject": "alice", "role": "admin"},
            "viewer-token-0123456789abcdefghijklmno": {"subject": "bob", "role": "viewer"},
        }
    )


@pytest.mark.parametrize(
    "header",
    ["Bearer abc.def-123", "bearer abc.def-123", "BEARER abc.def-123", "  Bearer   abc.def-123  "],
)
def test_bearer_scheme_is_case_insensitive_and_whitespace_tolerant(header: str) -> None:
    assert extract_bearer_token(header) == "abc.def-123"


@pytest.mark.parametrize("header", [None, "", "   "])
def test_missing_header(header: str | None) -> None:
    with pytest.raises(AuthenticationError) as exc_info:
        extract_bearer_token(header)
    assert exc_info.value.reason == "missing_token"
    assert exc_info.value.www_authenticate == 'Bearer realm="mcp-gateway"'


@pytest.mark.parametrize(
    "header",
    ["Basic dXNlcjpwYXNz", "Bearer", "Bearer ", "Token abc", "Bearer a b", "Bearer <script>"],
)
def test_malformed_header(header: str) -> None:
    with pytest.raises(AuthenticationError) as exc_info:
        extract_bearer_token(header)
    assert exc_info.value.reason == "malformed_header"
    assert 'error="invalid_token"' in exc_info.value.www_authenticate


def test_verify_known_tokens(verifier: StaticTokenVerifier) -> None:
    principal = verifier.verify("admin-token-0123456789abcdefghijklmnop")
    assert principal is not None
    assert (principal.subject, principal.role) == ("alice", "admin")
    assert principal.token_fingerprint == fingerprint("admin-token-0123456789abcdefghijklmnop")
    assert "admin-token" not in principal.token_fingerprint


@pytest.mark.parametrize(
    "token",
    ["admin-token-12345", "admin-token-0123456789abcdefghijklmnop7", "ADMIN-TOKEN-123456", "", "x"],
)
def test_verify_rejects_near_misses(verifier: StaticTokenVerifier, token: str) -> None:
    assert verifier.verify(token) is None


def test_authenticate_end_to_end(verifier: StaticTokenVerifier) -> None:
    assert authenticate("Bearer viewer-token-0123456789abcdefghijklmno", verifier).role == "viewer"
    with pytest.raises(AuthenticationError) as exc_info:
        authenticate("Bearer unknown-token-9999", verifier)
    assert exc_info.value.reason == "invalid_token"


def test_from_json_accepts_a_well_formed_document() -> None:
    good = json.dumps(
        {"tokens": {"long-enough-token-0123456789abcdefgh": {"subject": "s", "role": "r"}}}
    )
    assert (
        StaticTokenVerifier.from_json(good).verify("long-enough-token-0123456789abcdefgh")
        is not None
    )


@pytest.mark.parametrize(
    ("document", "match"),
    [
        (json.dumps({"tokens": {}}), "at least one token"),
        (json.dumps({"tokens": {"short": {"subject": "s", "role": "r"}}}), "at least 32"),
        (
            json.dumps({"tokens": {"long-enough-token-0123456789abcdefgh": {"subject": "s"}}}),
            "lacks a role",
        ),
        (
            json.dumps({"tokens": {"long-enough-token-0123456789abcdefgh": {"role": "r"}}}),
            "lacks a subject",
        ),
        (json.dumps({"nope": {}}), "tokens"),
        (json.dumps([]), "tokens"),
    ],
)
def test_from_json_rejects_malformed_documents(document: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        StaticTokenVerifier.from_json(document)


@pytest.mark.parametrize(
    ("document", "match"),
    [
        (
            '{"tokens": {"dup-token-0123456789abcdefghijklmn": {"subject": "s", "role": "r"}, '
            '"dup-token-0123456789abcdefghijklmn": {"subject": "t", "role": "admin"}}}',
            "duplicate member",
        ),
        (
            json.dumps(
                {"tokens": {"token with spaces 0123456789abcdef": {"subject": "s", "role": "r"}}}
            ),
            "token68",
        ),
        (
            json.dumps(
                {
                    "tokens": {
                        "token-0123456789abcdefghijklmnopqr": {"subject": "s\r\n", "role": "r"}
                    }
                }
            ),
            "subject",
        ),
        (
            json.dumps(
                {"tokens": {"token-0123456789abcdefghijklmnopqr": {"subject": "s", "role": " r"}}}
            ),
            "role",
        ),
        (
            json.dumps({"tokens": {"token-0123456789abcdefghijklmnopqr": "admin"}}),
            "identity must be an object",
        ),
    ],
)
def test_token_documents_are_validated_strictly(document: str, match: str) -> None:
    with pytest.raises(ValueError, match=match) as exc_info:
        StaticTokenVerifier.from_json(document)
    assert "0123456789abcdefghijklmn" not in str(exc_info.value)  # never quote a token


def test_development_documents_need_an_explicit_opt_in() -> None:
    document = json.dumps(
        {
            "development_only": True,
            "tokens": {"dev-only-token-0123456789abcdefghijklmn": {"subject": "s", "role": "r"}},
        }
    )
    with pytest.raises(ValueError, match="development_only"):
        StaticTokenVerifier.from_json(document)
    verifier = StaticTokenVerifier.from_json(document, allow_development=True)
    assert verifier.verify("dev-only-token-0123456789abcdefghijklmn") is not None


def test_repository_development_file_is_marked() -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "config" / "tokens.dev.json"
    with pytest.raises(ValueError, match="development_only"):
        StaticTokenVerifier.from_json_file(path)
    verifier = StaticTokenVerifier.from_json_file(path, allow_development=True)
    assert verifier.verify("dev-only-admin-token-not-for-production-0001") is not None


def test_fingerprints_are_keyed_per_process() -> None:
    import hashlib

    token = "admin-token-0123456789abcdefghijklmnop"
    assert fingerprint(token) == fingerprint(token)  # stable within the process
    assert fingerprint(token) != hashlib.sha256(token.encode()).hexdigest()[:12]
    assert len(fingerprint(token)) == 12
