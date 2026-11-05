"""Settings parsing and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_router.config import ConfigurationError, RouterSettings

A_KEY = "tenant-a-key-0123456789abcdefghijklmnop"
DUP_KEY = "dup-key-0123456789abcdefghijklmnopqrst"
KEY_ONE = "key-one-0123456789abcdefghijklmnopqr"
KEY_TWO = "key-two-0123456789abcdefghijklmnopqr"
BASE = {
    "MODEL_ROUTER_PRIMARY_URL": "http://127.0.0.1:9401/",
    "MODEL_ROUTER_SECONDARY_URL": "http://127.0.0.1:9402",
    "MODEL_ROUTER_TENANTS_JSON": json.dumps(
        {"tenants": {"tenant-a-key-0123456789abcdefghijklmnop": {"tenant_id": "tenant-a"}}}
    ),
}


def test_minimal_environment_parses_with_defaults() -> None:
    settings = RouterSettings.from_env(BASE)
    assert settings.primary.base_url == "http://127.0.0.1:9401"  # trailing slash removed
    assert settings.primary.timeout_ms == 3000
    assert settings.tokens_per_minute == 50_000
    assert settings.window_seconds == 60
    assert settings.tenants["tenant-a-key-0123456789abcdefghijklmnop"].tenant_id == "tenant-a"


def test_tenant_file_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "tenants.json"
    path.write_text(
        json.dumps(
            {
                "tenants": {
                    "file-key-0123456789abcdefghijklmnopqr": {
                        "tenant_id": "t",
                        "tokens_per_minute": 7,
                    }
                }
            }
        )
    )
    env = {k: v for k, v in BASE.items() if k != "MODEL_ROUTER_TENANTS_JSON"}
    settings = RouterSettings.from_env({**env, "MODEL_ROUTER_TENANTS_FILE": str(path)})
    assert settings.tenants["file-key-0123456789abcdefghijklmnopqr"].tokens_per_minute == 7


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"'
                + DUP_KEY
                + '": {"tenant_id": "a"}, "'
                + DUP_KEY
                + '": {"tenant_id": "b"}}}'
            },
            "duplicate member",
        ),
        ({"MODEL_ROUTER_TENANTS_JSON": "{not json"}, "not valid JSON"),
        ({"MODEL_ROUTER_TENANTS_JSON": '{"tenants": {}}'}, "non-empty"),
        (
            {"MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"short": {"tenant_id": "a"}}}'},
            "32 characters",
        ),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"long-enough-key-0123456789abcdefghij": '
                '{"tenant_id": "a", "tokens_per_minute": 0}}}'
            },
            "positive int",
        ),
        ({"MODEL_ROUTER_PRIMARY_TIMEOUT_MS": "0"}, "timeout_ms must be positive"),
        ({"MODEL_ROUTER_PRIMARY_TIMEOUT_MS": "soon"}, "invalid setting"),
        ({"MODEL_ROUTER_SECONDARY_URL": "127.0.0.1:9402"}, "absolute http"),
        ({"MODEL_ROUTER_SECONDARY_NAME": "primary"}, "different names"),
        ({"MODEL_ROUTER_TOKENS_PER_MINUTE": "-1"}, "must be positive"),
        ({"MODEL_ROUTER_LOG_LEVEL": "loud"}, "log_level"),
        ({"MODEL_ROUTER_CIRCUIT_FAILURE_THRESHOLD": "0"}, "at least 1"),
        ({"MODEL_ROUTER_CIRCUIT_RECOVERY_SECONDS": "nan"}, "positive finite"),
        ({"MODEL_ROUTER_EVICTION_INTERVAL_SECONDS": "0"}, "positive finite"),
        ({"MODEL_ROUTER_MAX_BODY_BYTES": "0"}, "max_body_bytes"),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": json.dumps(
                    {
                        "tenants": {
                            KEY_ONE: {"tenant_id": "t", "tokens_per_minute": 10},
                            KEY_TWO: {"tenant_id": "t", "tokens_per_minute": 20},
                        }
                    }
                )
            },
            "disagree",
        ),
    ],
)
def test_invalid_environments_are_rejected(override: dict[str, str], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        RouterSettings.from_env({**BASE, **override})


def test_missing_tenant_file_is_a_configuration_error(tmp_path: Path) -> None:
    env = {k: v for k, v in BASE.items() if k != "MODEL_ROUTER_TENANTS_JSON"}
    with pytest.raises(ConfigurationError, match="cannot read"):
        RouterSettings.from_env(
            {**env, "MODEL_ROUTER_TENANTS_FILE": str(tmp_path / "missing.json")}
        )


def test_no_tenant_source_is_a_configuration_error() -> None:
    env = {k: v for k, v in BASE.items() if k != "MODEL_ROUTER_TENANTS_JSON"}
    with pytest.raises(ConfigurationError, match="TENANTS_FILE"):
        RouterSettings.from_env(env)


def test_failover_statuses_include_the_provider_request_timeout() -> None:
    assert 408 in RouterSettings.from_env(BASE).failover_statuses
    assert RouterSettings.from_env(BASE).log_level == "INFO"
    assert RouterSettings.from_env({**BASE, "MODEL_ROUTER_LOG_LEVEL": "debug"}).log_level == "DEBUG"


def test_provider_dialects_concurrency_and_fallback_settings_parse() -> None:
    settings = RouterSettings.from_env(
        {
            **BASE,
            "MODEL_ROUTER_SECONDARY_DIALECT": "Anthropic",
            "MODEL_ROUTER_PRIMARY_MAX_CONCURRENCY": "8",
            "MODEL_ROUTER_ALLOW_FALLBACK_DEFAULT": "no",
            "MODEL_ROUTER_TOKEN_ESTIMATOR": "heuristic:3.5",
            "MODEL_ROUTER_MAX_COOLDOWN_SECONDS": "15",
        }
    )
    assert settings.secondary.dialect == "anthropic"
    assert settings.primary.dialect == "openai"
    assert settings.primary.max_concurrency == 8
    assert settings.allow_fallback_default is False
    assert settings.token_estimator == "heuristic:3.5"
    assert settings.max_cooldown_seconds == 15.0
    assert 529 in settings.failover_statuses


def test_tenant_models_parse_and_validate() -> None:
    document = {
        "tenants": {
            "tenant-a-key-0123456789abcdefghijklmnop": {
                "tenant_id": "a",
                "model": "a-ft",
                "allowed_models": ["a-ft", "general"],
            }
        }
    }
    settings = RouterSettings.from_env({**BASE, "MODEL_ROUTER_TENANTS_JSON": json.dumps(document)})
    tenant = settings.tenants["tenant-a-key-0123456789abcdefghijklmnop"]
    assert (tenant.model, tenant.allowed_models) == ("a-ft", ("a-ft", "general"))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"MODEL_ROUTER_SECONDARY_DIALECT": "cohere"}, "dialect"),
        ({"MODEL_ROUTER_PRIMARY_URL": "http://user:pw@127.0.0.1:9401"}, "credentials"),
        ({"MODEL_ROUTER_PRIMARY_MODEL": "bad model"}, "model must be"),
        ({"MODEL_ROUTER_PRIMARY_MODEL": "m" * 257}, "model must be"),
        ({"MODEL_ROUTER_PRIMARY_MAX_TOKENS_FIELD": "tokens"}, "max_tokens_field"),
        ({"MODEL_ROUTER_MAX_RESPONSE_BYTES": "0"}, "max_response_bytes"),
        ({"MODEL_ROUTER_ALLOW_DEVELOPMENT_CREDENTIALS": "sure"}, "true or false"),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"' + A_KEY + '": '
                '{"tenant_id": "a", "model": "a b"}}}'
            },
            "model must be",
        ),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"' + A_KEY + '": '
                '{"tenant_id": "a", "allowed_models": ["ok", "bad\\nname"]}}}'
            },
            "allowed_models",
        ),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"' + A_KEY + '": '
                '{"tenant_id": " padded "}}}'
            },
            "tenant_id",
        ),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"tenant key with spaces 0123456789": '
                '{"tenant_id": "a"}}}'
            },
            "printable ASCII",
        ),
        ({"MODEL_ROUTER_PRIMARY_MAX_CONCURRENCY": "-1"}, "max_concurrency"),
        ({"MODEL_ROUTER_ALLOW_FALLBACK_DEFAULT": "maybe"}, "true or false"),
        ({"MODEL_ROUTER_MAX_COOLDOWN_SECONDS": "0"}, "max_cooldown_seconds"),
        ({"MODEL_ROUTER_TOKEN_ESTIMATOR": " "}, "token_estimator"),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"' + A_KEY + '": '
                '{"tenant_id": "a", "model": "x", "allowed_models": ["y"]}}}'
            },
            "not in allowed_models",
        ),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"' + A_KEY + '": '
                '{"tenant_id": "a", "model": ""}}}'
            },
            "non-empty string",
        ),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"' + A_KEY + '": '
                '{"tenant_id": "a", "allowed_models": "general"}}}'
            },
            "allowed_models",
        ),
        (
            {
                "MODEL_ROUTER_TENANTS_JSON": json.dumps(
                    {
                        "tenants": {
                            KEY_ONE: {"tenant_id": "t", "model": "m1"},
                            KEY_TWO: {"tenant_id": "t", "model": "m2"},
                        }
                    }
                )
            },
            "disagree",
        ),
    ],
)
def test_new_settings_are_validated(override: dict[str, str], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        RouterSettings.from_env({**BASE, **override})


def test_configuration_errors_never_quote_key_material() -> None:
    document = (
        '{"tenants": {"dup-key-0123456789abcdefghijklmnopqrst": {"tenant_id": "a"}, '
        '"dup-key-0123456789abcdefghijklmnopqrst": {"tenant_id": "b"}}}'
    )
    with pytest.raises(ConfigurationError) as duplicate:
        RouterSettings.from_env({**BASE, "MODEL_ROUTER_TENANTS_JSON": document})
    assert "dup-key" not in str(duplicate.value)
    with pytest.raises(ConfigurationError) as missing:
        RouterSettings.from_env(
            {
                **BASE,
                "MODEL_ROUTER_TENANTS_JSON": '{"tenants": {"secret-key-0123456789abcdefghij": {}}}',
            }
        )
    assert "secret-key" not in str(missing.value)
    assert "entry 1" in str(missing.value)


def test_development_credentials_need_an_explicit_opt_in() -> None:
    document = json.dumps(
        {
            "development_only": True,
            "tenants": {"dev-only-tenant-key-0123456789abcdefghij": {"tenant_id": "dev"}},
        }
    )
    with pytest.raises(ConfigurationError, match="development_only"):
        RouterSettings.from_env({**BASE, "MODEL_ROUTER_TENANTS_JSON": document})
    settings = RouterSettings.from_env(
        {
            **BASE,
            "MODEL_ROUTER_TENANTS_JSON": document,
            "MODEL_ROUTER_ALLOW_DEVELOPMENT_CREDENTIALS": "true",
        }
    )
    assert settings.tenants["dev-only-tenant-key-0123456789abcdefghij"].tenant_id == "dev"


def test_repository_development_file_is_marked_and_loads_with_the_flag() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "tenants.dev.json"
    env = {k: v for k, v in BASE.items() if k != "MODEL_ROUTER_TENANTS_JSON"}
    with pytest.raises(ConfigurationError, match="development_only"):
        RouterSettings.from_env({**env, "MODEL_ROUTER_TENANTS_FILE": str(path)})
    settings = RouterSettings.from_env(
        {
            **env,
            "MODEL_ROUTER_TENANTS_FILE": str(path),
            "MODEL_ROUTER_ALLOW_DEVELOPMENT_CREDENTIALS": "true",
        }
    )
    assert {t.tenant_id for t in settings.tenants.values()} == {"tenant-a", "tenant-b"}


def test_provider_max_tokens_field_and_response_cap_parse() -> None:
    settings = RouterSettings.from_env(
        {
            **BASE,
            "MODEL_ROUTER_PRIMARY_MAX_TOKENS_FIELD": "max_completion_tokens",
            "MODEL_ROUTER_MAX_RESPONSE_BYTES": "4096",
        }
    )
    assert settings.primary.max_tokens_field == "max_completion_tokens"
    assert settings.secondary.max_tokens_field == "max_tokens"
    assert settings.max_response_bytes == 4096
