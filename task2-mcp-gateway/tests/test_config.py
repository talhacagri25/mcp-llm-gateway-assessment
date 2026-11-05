"""Settings parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_gateway.config import ConfigurationError, GatewaySettings

BASE = {
    "MCP_GATEWAY_DOWNSTREAM_URL": "http://127.0.0.1:9100/mcp",
    "MCP_GATEWAY_TOKENS_JSON": '{"tokens": {}}',
}


def test_defaults_from_minimal_environment() -> None:
    settings = GatewaySettings.from_env(BASE)
    assert settings.mcp_path == "/mcp"
    assert settings.admin_tool_prefixes == ("admin_",)
    assert settings.admin_roles == ("admin",)
    assert settings.denied_http_status == 200
    assert settings.max_body_bytes == 1024 * 1024


def test_overrides_are_parsed() -> None:
    settings = GatewaySettings.from_env(
        {
            **BASE,
            "MCP_GATEWAY_TOKENS_FILE": "/etc/gateway/tokens.json",
            "MCP_GATEWAY_PATH": "/v1/mcp",
            "MCP_GATEWAY_ADMIN_TOOL_PREFIXES": "admin_, danger_ ,",
            "MCP_GATEWAY_ADMIN_ROLES": "admin,sre",
            "MCP_GATEWAY_DENIED_HTTP_STATUS": "403",
            "MCP_GATEWAY_MAX_BODY_BYTES": "2048",
            "MCP_GATEWAY_READ_TIMEOUT_SECONDS": "2.5",
            "MCP_GATEWAY_STREAM_READ_TIMEOUT_SECONDS": "600",
            "MCP_GATEWAY_DOWNSTREAM_TOKEN": "svc-token",
        }
    )
    assert settings.tokens_file == Path("/etc/gateway/tokens.json")
    assert settings.mcp_path == "/v1/mcp"
    assert settings.admin_tool_prefixes == ("admin_", "danger_")
    assert settings.admin_roles == ("admin", "sre")
    assert settings.denied_http_status == 403
    assert settings.max_body_bytes == 2048
    assert settings.read_timeout_seconds == 2.5
    assert settings.stream_read_timeout_seconds == 600.0
    assert settings.downstream_service_token == "svc-token"


def test_capacity_settings_are_parsed() -> None:
    settings = GatewaySettings.from_env(
        {
            **BASE,
            "MCP_GATEWAY_MAX_CONNECTIONS": "50",
            "MCP_GATEWAY_MAX_INFLIGHT_PER_SUBJECT": "0",
            "MCP_GATEWAY_MAX_BATCH_SIZE": "10",
            "MCP_GATEWAY_LOG_LEVEL": "debug",
        }
    )
    assert settings.max_connections == 50
    assert settings.max_inflight_per_subject == 0
    assert settings.max_batch_size == 10
    assert settings.log_level == "DEBUG"
    defaults = GatewaySettings.from_env(BASE)
    assert (defaults.max_connections, defaults.max_inflight_per_subject) == (1000, 64)
    assert defaults.max_batch_size == 100


def test_stream_timeout_zero_means_unlimited() -> None:
    settings = GatewaySettings.from_env({**BASE, "MCP_GATEWAY_STREAM_READ_TIMEOUT_SECONDS": "0"})
    assert settings.stream_read_timeout_seconds is None
    assert GatewaySettings.from_env(BASE).stream_read_timeout_seconds == 300.0


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"MCP_GATEWAY_DOWNSTREAM_URL": "http://127.0.0.1:9100/mcp"},  # no token source
        {**BASE, "MCP_GATEWAY_DOWNSTREAM_URL": "127.0.0.1:9100"},
        {**BASE, "MCP_GATEWAY_DOWNSTREAM_URL": "ftp://x/mcp"},
        {**BASE, "MCP_GATEWAY_DENIED_HTTP_STATUS": "401"},
        {**BASE, "MCP_GATEWAY_DENIED_HTTP_STATUS": "abc"},
        {**BASE, "MCP_GATEWAY_PATH": "mcp"},
        {**BASE, "MCP_GATEWAY_MAX_BODY_BYTES": "0"},
        {**BASE, "MCP_GATEWAY_ADMIN_ROLES": ","},
        {**BASE, "MCP_GATEWAY_READ_TIMEOUT_SECONDS": "0"},
        {**BASE, "MCP_GATEWAY_STREAM_READ_TIMEOUT_SECONDS": "-1"},
        {**BASE, "MCP_GATEWAY_STREAM_READ_TIMEOUT_SECONDS": "soon"},
        {**BASE, "MCP_GATEWAY_READ_TIMEOUT_SECONDS": "nan"},
        {**BASE, "MCP_GATEWAY_CONNECT_TIMEOUT_SECONDS": "inf"},
        {**BASE, "MCP_GATEWAY_LOG_LEVEL": "verbose"},
        {**BASE, "MCP_GATEWAY_PATH": "/healthz"},
        {**BASE, "MCP_GATEWAY_MAX_BATCH_SIZE": "0"},
        {**BASE, "MCP_GATEWAY_MAX_CONNECTIONS": "0"},
        {**BASE, "MCP_GATEWAY_MAX_INFLIGHT_PER_SUBJECT": "-1"},
    ],
)
def test_invalid_environments_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        GatewaySettings.from_env(env)


def test_credentialed_urls_and_odd_service_tokens_are_refused() -> None:
    from mcp_gateway.config import ConfigurationError, GatewaySettings

    with pytest.raises(ConfigurationError, match="credentials"):
        GatewaySettings(downstream_url="http://user:pw@downstream/mcp", tokens_json="{}")
    with pytest.raises(ConfigurationError, match="DOWNSTREAM_TOKEN"):
        GatewaySettings(
            downstream_url="http://downstream/mcp",
            tokens_json="{}",
            downstream_service_token="bad token\r\n",
        )
    settings = GatewaySettings.from_env(
        {
            "MCP_GATEWAY_DOWNSTREAM_URL": "http://downstream/mcp",
            "MCP_GATEWAY_TOKENS_JSON": "{}",
            "MCP_GATEWAY_ALLOW_DEVELOPMENT_CREDENTIALS": "yes",
        }
    )
    assert settings.allow_development_credentials is True
    with pytest.raises(ConfigurationError, match="true or false"):
        GatewaySettings.from_env(
            {
                "MCP_GATEWAY_DOWNSTREAM_URL": "http://downstream/mcp",
                "MCP_GATEWAY_TOKENS_JSON": "{}",
                "MCP_GATEWAY_ALLOW_DEVELOPMENT_CREDENTIALS": "sometimes",
            }
        )
