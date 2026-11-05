"""Settings parsing, pattern selection and the declared detector file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_guardrail_gateway.config import ConfigurationError, GuardrailSettings

BASE = {"LLM_GATEWAY_UPSTREAM_BASE_URL": "http://127.0.0.1:9200/"}
KEYED = {**BASE, "LLM_GATEWAY_UPSTREAM_API_KEY": "sk-upstream"}


def test_defaults() -> None:
    settings = GuardrailSettings.from_env(BASE)
    assert settings.upstream_base_url == "http://127.0.0.1:9200"
    assert settings.pattern_names == ("email", "credit_card", "ssn")
    assert [p.name for p in settings.patterns] == ["email", "credit_card", "ssn"]
    assert settings.log_level == "INFO"


def test_presets_are_enabled_by_name() -> None:
    settings = GuardrailSettings.from_env({**BASE, "LLM_GATEWAY_PATTERNS": "email,iban,phone_us"})
    assert [p.name for p in settings.patterns] == ["email", "iban", "phone_us"]


def test_pattern_file_adds_detectors_enabled_by_default(tmp_path: Path) -> None:
    path = tmp_path / "patterns.json"
    path.write_text(
        json.dumps(
            {
                "patterns": [
                    {
                        "name": "customer_id",
                        "regex": "CUST-[0-9]{5}",
                        "alphabet": "CUST-0123456789",
                        "start_chars": "C",
                        "max_length": 10,
                    }
                ]
            }
        )
    )
    settings = GuardrailSettings.from_env({**BASE, "LLM_GATEWAY_PATTERNS_FILE": str(path)})
    assert settings.pattern_names == ("email", "credit_card", "ssn", "customer_id")
    subset = GuardrailSettings.from_env(
        {**BASE, "LLM_GATEWAY_PATTERNS_FILE": str(path), "LLM_GATEWAY_PATTERNS": "customer_id"}
    )
    assert [p.name for p in subset.patterns] == ["customer_id"]


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"LLM_GATEWAY_PATTERNS": "email,unknown"}, "must name one or more"),
        ({"LLM_GATEWAY_PATTERNS": "email,email"}, "twice"),
        ({"LLM_GATEWAY_PATTERNS": ","}, "must name one or more"),
        ({"LLM_GATEWAY_LOG_LEVEL": "loud"}, "LOG_LEVEL"),
        ({"LLM_GATEWAY_READ_TIMEOUT_SECONDS": "nan"}, "positive finite"),
        ({"LLM_GATEWAY_MAX_HOLDBACK_CHARS": "4"}, "at least 16"),
        ({"LLM_GATEWAY_PATTERNS_FILE": "/nonexistent/patterns.json"}, "cannot read"),
        ({"LLM_GATEWAY_UPSTREAM_BASE_URL": "127.0.0.1:9200"}, "absolute http"),
    ],
)
def test_invalid_environments_are_rejected(override: dict[str, str], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        GuardrailSettings.from_env({**BASE, **override})


def test_malformed_pattern_file_is_a_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "patterns.json"
    path.write_text('{"patterns": [{"name": "x"}]}')
    with pytest.raises(ConfigurationError, match="regex is required"):
        GuardrailSettings.from_env({**BASE, "LLM_GATEWAY_PATTERNS_FILE": str(path)})
    path.write_text("{not json")
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        GuardrailSettings.from_env({**BASE, "LLM_GATEWAY_PATTERNS_FILE": str(path)})


def test_open_relay_on_the_operator_key_needs_an_explicit_flag() -> None:
    with pytest.raises(ConfigurationError, match="ALLOW_UNAUTHENTICATED"):
        GuardrailSettings.from_env(KEYED)
    keyed = GuardrailSettings.from_env({**KEYED, "LLM_GATEWAY_API_KEYS": "gw-1"})
    assert keyed.api_keys == ("gw-1",)
    opened = GuardrailSettings.from_env({**KEYED, "LLM_GATEWAY_ALLOW_UNAUTHENTICATED": "true"})
    assert opened.allow_unauthenticated is True
    # Bring your own key mode (no upstream key) stays allowed without the flag.
    assert GuardrailSettings.from_env(BASE).upstream_api_key is None


def test_credentials_in_the_upstream_url_are_refused() -> None:
    with pytest.raises(ConfigurationError, match="embed credentials"):
        GuardrailSettings.from_env(
            {"LLM_GATEWAY_UPSTREAM_BASE_URL": "https://svc:secret@provider.example"}
        )


def test_stream_budgets_are_validated() -> None:
    assert GuardrailSettings.from_env(BASE).max_stream_seconds == 600.0
    with pytest.raises(ConfigurationError, match="max_stream_seconds"):
        GuardrailSettings.from_env({**BASE, "LLM_GATEWAY_MAX_STREAM_SECONDS": "0"})
    with pytest.raises(ConfigurationError, match="max_stream_bytes"):
        GuardrailSettings.from_env({**BASE, "LLM_GATEWAY_MAX_STREAM_BYTES": "-1"})
