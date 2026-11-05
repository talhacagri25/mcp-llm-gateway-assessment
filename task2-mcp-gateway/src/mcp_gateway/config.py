"""Gateway settings, sourced from environment variables.

Every value has a safe default except the two that cannot: the downstream
URL and the token source. The gateway refuses to start without them rather
than falling back to an open configuration.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ENV_PREFIX = "MCP_GATEWAY_"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
HEALTH_PATH = "/healthz"
_TOKEN68_RE = re.compile(r"[A-Za-z0-9\-._~+/]+=*")


class ConfigurationError(ValueError):
    """Raised when the environment does not describe a usable gateway."""


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_bool(value: str, *, name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false, got {value!r}")


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    downstream_url: str
    tokens_file: Path | None = None
    tokens_json: str | None = None
    mcp_path: str = "/mcp"
    admin_tool_prefixes: tuple[str, ...] = ("admin_",)
    admin_roles: tuple[str, ...] = ("admin",)
    denied_http_status: int = 200
    max_body_bytes: int = 1024 * 1024
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    """Maximum silence tolerated while reading a POST response."""
    stream_read_timeout_seconds: float | None = 300.0
    """Maximum silence tolerated on a long-lived GET event stream; ``None`` means unlimited.

    The official SDK sends an SSE keep-alive comment every 15 seconds, so a
    healthy stream never comes close to this bound; it exists to reclaim the
    connection when the downstream dies silently.
    """
    downstream_service_token: str | None = None
    max_connections: int = 1000
    """Size of the downstream connection pool (each open GET event stream holds one)."""
    max_inflight_per_subject: int = 64
    """Concurrent requests one authenticated subject may have open; 0 disables the cap."""
    max_batch_size: int = 100
    """Maximum number of messages in one JSON-RPC batch."""
    allow_development_credentials: bool = False
    """Accept a token document marked ``development_only`` (the repository's example file)."""
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        parts = urlsplit(self.downstream_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ConfigurationError(
                f"{ENV_PREFIX}DOWNSTREAM_URL must be an absolute http(s) URL, "
                f"got {self.downstream_url!r}"
            )
        if parts.username or parts.password:
            raise ConfigurationError(
                f"{ENV_PREFIX}DOWNSTREAM_URL must not embed credentials; "
                f"use {ENV_PREFIX}DOWNSTREAM_TOKEN"
            )
        if self.downstream_service_token is not None and not _TOKEN68_RE.fullmatch(
            self.downstream_service_token
        ):
            raise ConfigurationError(
                f"{ENV_PREFIX}DOWNSTREAM_TOKEN must be a bearer token "
                "(letters, digits, - . _ ~ + / and trailing =)"
            )
        if self.tokens_file is None and self.tokens_json is None:
            raise ConfigurationError(
                f"set {ENV_PREFIX}TOKENS_FILE or {ENV_PREFIX}TOKENS_JSON; the gateway does not "
                "start without a token source"
            )
        if self.denied_http_status not in {200, 403}:
            raise ConfigurationError(f"{ENV_PREFIX}DENIED_HTTP_STATUS must be 200 or 403")
        if not self.mcp_path.startswith("/"):
            raise ConfigurationError(f"{ENV_PREFIX}PATH must start with '/'")
        if not self.admin_tool_prefixes or not self.admin_roles:
            raise ConfigurationError("admin tool prefixes and admin roles must not be empty")
        if self.max_body_bytes <= 0:
            raise ConfigurationError(f"{ENV_PREFIX}MAX_BODY_BYTES must be positive")
        for name in ("connect_timeout_seconds", "read_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive finite number")
        stream = self.stream_read_timeout_seconds
        if stream is not None and (not math.isfinite(stream) or stream <= 0):
            raise ConfigurationError(
                f"{ENV_PREFIX}STREAM_READ_TIMEOUT_SECONDS must be positive, or 0 for no limit"
            )
        if self.max_connections <= 0 or self.max_batch_size <= 0:
            raise ConfigurationError("max_connections and max_batch_size must be positive")
        if self.max_inflight_per_subject < 0:
            raise ConfigurationError(f"{ENV_PREFIX}MAX_INFLIGHT_PER_SUBJECT must be 0 or more")
        if self.log_level.upper() not in LOG_LEVELS:
            raise ConfigurationError(
                f"{ENV_PREFIX}LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}"
            )
        if self.mcp_path.rstrip("/") == HEALTH_PATH:
            raise ConfigurationError(f"{ENV_PREFIX}PATH must not shadow the {HEALTH_PATH} probe")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GatewaySettings:
        source = os.environ if env is None else env

        def get(name: str, default: str | None = None) -> str | None:
            value = source.get(ENV_PREFIX + name)
            return value if value not in (None, "") else default

        downstream_url = get("DOWNSTREAM_URL")
        if downstream_url is None:
            raise ConfigurationError(f"{ENV_PREFIX}DOWNSTREAM_URL is required")
        tokens_file = get("TOKENS_FILE")
        try:
            stream_timeout = float(get("STREAM_READ_TIMEOUT_SECONDS", "300") or "300")
            return cls(
                downstream_url=downstream_url,
                tokens_file=Path(tokens_file) if tokens_file else None,
                tokens_json=get("TOKENS_JSON"),
                mcp_path=get("PATH", "/mcp") or "/mcp",
                admin_tool_prefixes=_csv(get("ADMIN_TOOL_PREFIXES", "admin_") or ""),
                admin_roles=_csv(get("ADMIN_ROLES", "admin") or ""),
                denied_http_status=int(get("DENIED_HTTP_STATUS", "200") or "200"),
                max_body_bytes=int(get("MAX_BODY_BYTES", str(1024 * 1024)) or "0"),
                connect_timeout_seconds=float(get("CONNECT_TIMEOUT_SECONDS", "5") or "5"),
                read_timeout_seconds=float(get("READ_TIMEOUT_SECONDS", "30") or "30"),
                stream_read_timeout_seconds=None if stream_timeout == 0 else stream_timeout,
                downstream_service_token=get("DOWNSTREAM_TOKEN"),
                max_connections=int(get("MAX_CONNECTIONS", "1000") or "1000"),
                max_inflight_per_subject=int(get("MAX_INFLIGHT_PER_SUBJECT", "64") or "64"),
                max_batch_size=int(get("MAX_BATCH_SIZE", "100") or "100"),
                allow_development_credentials=_parse_bool(
                    get("ALLOW_DEVELOPMENT_CREDENTIALS", "false") or "false",
                    name=f"{ENV_PREFIX}ALLOW_DEVELOPMENT_CREDENTIALS",
                ),
                log_level=(get("LOG_LEVEL", "INFO") or "INFO").upper(),
            )
        except ValueError as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(f"invalid numeric setting: {exc}") from exc
