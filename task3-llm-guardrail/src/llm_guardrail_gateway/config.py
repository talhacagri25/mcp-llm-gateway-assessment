"""Gateway settings sourced from environment variables."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from llm_guardrail_gateway.redaction.custom import PatternConfigurationError, load_pattern_file
from llm_guardrail_gateway.redaction.patterns import (
    DEFAULT_PATTERN_NAMES,
    PATTERNS_BY_NAME,
    PiiPattern,
)

ENV_PREFIX = "LLM_GATEWAY_"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigurationError(ValueError):
    pass


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class GuardrailSettings:
    upstream_base_url: str
    upstream_api_key: str | None = None
    api_keys: tuple[str, ...] = field(default_factory=tuple)
    """Keys clients must present to the gateway. Empty means the gateway is open (local use)."""
    pattern_names: tuple[str, ...] = DEFAULT_PATTERN_NAMES
    """Enabled detectors: built-in names, preset names or names declared in the pattern file."""
    custom_patterns: tuple[PiiPattern, ...] = ()
    """Deployment specific detectors from ``LLM_GATEWAY_PATTERNS_FILE``, enabled by name."""
    replacement: str = "[REDACTED]"
    max_holdback_chars: int = 256
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 120.0
    """Maximum silence between two upstream chunks before the stream is abandoned."""
    max_body_bytes: int = 1024 * 1024
    max_event_bytes: int = 256 * 1024
    max_response_bytes: int = 8 * 1024 * 1024
    max_stream_seconds: float = 600.0
    """Longest a single streamed response may run before the gateway ends it."""
    max_stream_bytes: int = 64 * 1024 * 1024
    """Most bytes one streamed response may deliver before the gateway ends it."""
    allow_unauthenticated: bool = False
    """Permit an open gateway that injects the upstream key for anonymous callers.
    Off by default: that configuration is an open relay on the operator's credential."""
    forward_headers: tuple[str, ...] = ("traceparent", "tracestate")
    """Additional request headers relayed to the upstream (lower case)."""
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        parts = urlsplit(self.upstream_base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ConfigurationError(
                f"{ENV_PREFIX}UPSTREAM_BASE_URL must be an absolute http(s) URL"
            )
        custom_names = {p.name for p in self.custom_patterns}
        available = set(PATTERNS_BY_NAME) | custom_names
        unknown = set(self.pattern_names) - available
        if unknown or not self.pattern_names:
            raise ConfigurationError(
                f"{ENV_PREFIX}PATTERNS must name one or more of {sorted(available)}"
            )
        if len(set(self.pattern_names)) != len(self.pattern_names):
            raise ConfigurationError(f"{ENV_PREFIX}PATTERNS lists a detector twice")
        if not self.replacement:
            raise ConfigurationError(f"{ENV_PREFIX}REDACTION_TOKEN must not be empty")
        if self.max_holdback_chars < 16:
            raise ConfigurationError(f"{ENV_PREFIX}MAX_HOLDBACK_CHARS must be at least 16")
        for name in ("max_body_bytes", "max_event_bytes", "max_response_bytes"):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"{name} must be positive")
        for name in ("connect_timeout_seconds", "read_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive finite number")
        if self.log_level.upper() not in LOG_LEVELS:
            raise ConfigurationError(
                f"{ENV_PREFIX}LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}"
            )
        if parts.username or parts.password:
            raise ConfigurationError(
                f"{ENV_PREFIX}UPSTREAM_BASE_URL must not embed credentials; "
                f"use {ENV_PREFIX}UPSTREAM_API_KEY"
            )
        if not math.isfinite(self.max_stream_seconds) or self.max_stream_seconds <= 0:
            raise ConfigurationError("max_stream_seconds must be a positive finite number")
        if self.max_stream_bytes <= 0:
            raise ConfigurationError("max_stream_bytes must be positive")
        if self.upstream_api_key and not self.api_keys and not self.allow_unauthenticated:
            raise ConfigurationError(
                f"set {ENV_PREFIX}API_KEYS, or set {ENV_PREFIX}ALLOW_UNAUTHENTICATED=true to run "
                "an open gateway that spends the upstream key for anonymous callers"
            )

    @property
    def patterns(self) -> tuple[PiiPattern, ...]:
        """The enabled detectors, in configuration order."""
        by_name = {**PATTERNS_BY_NAME, **{p.name: p for p in self.custom_patterns}}
        return tuple(by_name[name] for name in self.pattern_names)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GuardrailSettings:
        source = os.environ if env is None else env

        def get(name: str, default: str) -> str:
            value = source.get(ENV_PREFIX + name)
            return value if value not in (None, "") else default

        upstream = source.get(ENV_PREFIX + "UPSTREAM_BASE_URL")
        if not upstream:
            raise ConfigurationError(f"{ENV_PREFIX}UPSTREAM_BASE_URL is required")
        patterns_file = get("PATTERNS_FILE", "")
        custom: tuple[PiiPattern, ...] = ()
        if patterns_file:
            try:
                custom = load_pattern_file(Path(patterns_file))
            except PatternConfigurationError as exc:
                raise ConfigurationError(f"{ENV_PREFIX}PATTERNS_FILE: {exc}") from exc
        default_names = ",".join((*DEFAULT_PATTERN_NAMES, *(p.name for p in custom)))
        try:
            return cls(
                upstream_base_url=upstream.rstrip("/"),
                upstream_api_key=source.get(ENV_PREFIX + "UPSTREAM_API_KEY") or None,
                api_keys=_csv(get("API_KEYS", "")),
                pattern_names=_csv(get("PATTERNS", default_names)),
                custom_patterns=custom,
                replacement=get("REDACTION_TOKEN", "[REDACTED]"),
                max_holdback_chars=int(get("MAX_HOLDBACK_CHARS", "256")),
                connect_timeout_seconds=float(get("CONNECT_TIMEOUT_SECONDS", "5")),
                read_timeout_seconds=float(get("READ_TIMEOUT_SECONDS", "120")),
                max_body_bytes=int(get("MAX_BODY_BYTES", str(1024 * 1024))),
                max_event_bytes=int(get("MAX_EVENT_BYTES", str(256 * 1024))),
                max_response_bytes=int(get("MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))),
                max_stream_seconds=float(get("MAX_STREAM_SECONDS", "600")),
                max_stream_bytes=int(get("MAX_STREAM_BYTES", str(64 * 1024 * 1024))),
                allow_unauthenticated=get("ALLOW_UNAUTHENTICATED", "false").strip().lower()
                in {"1", "true", "yes", "on"},
                forward_headers=tuple(
                    h.lower() for h in _csv(get("FORWARD_HEADERS", "traceparent,tracestate"))
                ),
                log_level=get("LOG_LEVEL", "INFO").upper(),
            )
        except ValueError as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(f"invalid numeric setting: {exc}") from exc
