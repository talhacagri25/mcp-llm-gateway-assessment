"""Router settings sourced from environment variables."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from model_router.dialects import (
    DEFAULT_FORWARDED_CLIENT_HEADERS,
    DIALECT_NAMES,
    MAX_TOKENS_FIELDS,
)

ENV_PREFIX = "MODEL_ROUTER_"
DEFAULT_FAILOVER_STATUSES = (408, 429, 500, 502, 503, 504, 529)
"""Provider statuses that trigger failover: the provider's own request timeout (408),
rate limits (429), server errors and Anthropic's overloaded signal (529)."""
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_NEVER_FORWARDED = frozenset(
    {
        "authorization",
        "cookie",
        "x-api-key",
        "x-goog-api-key",
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "x-request-id",
    }
)
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
MIN_API_KEY_LENGTH: Final = 32
"""Tenant keys are bearer secrets; ``secrets.token_urlsafe(32)`` is the suggested generator."""
MODEL_NAME_RE: Final = re.compile(r"[A-Za-z0-9._:/@+-]{1,256}")
"""Model identifiers across providers (``ft:gpt-4o:org::id``, ``org/model``, ``@cf/meta/x``),
bounded so a value is safe in a response header, a log line and a ledger row."""
MODEL_NAME_RULE: Final = "1 to 256 characters from letters, digits and . _ : / @ + -"
DEVELOPMENT_MARKER: Final = "development_only"
"""Top-level flag in a tenant document whose keys are published (the repository's
development file); such a document is refused unless explicitly allowed."""
_TENANT_ID_RE: Final = re.compile(r"[\x21-\x7e][\x20-\x7e]{0,126}[\x21-\x7e]|[\x21-\x7e]")


class ConfigurationError(ValueError):
    pass


def parse_bool(value: str, *, name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigurationError(f"{name} must be true or false, got {value!r}")


def valid_model_name(value: Any) -> bool:
    return isinstance(value, str) and MODEL_NAME_RE.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    name: str
    base_url: str
    api_key: str | None = None
    model: str | None = None
    """Model identifier to send to this provider (overrides the client's value when set)."""
    timeout_ms: int = 3000
    dialect: str = "openai"
    """``openai`` (Chat Completions compatible) or ``anthropic`` (Messages API, translated)."""
    max_concurrency: int = 0
    """Requests in flight to this provider at once before the router fails over; 0 = unlimited.
    The right knob for a self-hosted model whose capacity is GPU throughput, not a quota."""
    max_tokens_field: str = "max_tokens"
    """Field the ``openai`` dialect fills with the default completion budget when the client
    sent none (``max_completion_tokens`` for providers that retired ``max_tokens``)."""

    def __post_init__(self) -> None:
        parts = urlsplit(self.base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ConfigurationError(
                f"provider {self.name!r}: base_url must be an absolute http(s) URL"
            )
        if parts.username or parts.password:
            raise ConfigurationError(
                f"provider {self.name!r}: base_url must not embed credentials; use the API key"
            )
        if self.timeout_ms <= 0:
            raise ConfigurationError(f"provider {self.name!r}: timeout_ms must be positive")
        if self.dialect not in DIALECT_NAMES:
            raise ConfigurationError(
                f"provider {self.name!r}: dialect must be one of {', '.join(DIALECT_NAMES)}"
            )
        if self.max_concurrency < 0:
            raise ConfigurationError(f"provider {self.name!r}: max_concurrency must be 0 or more")
        if self.model is not None and not valid_model_name(self.model):
            raise ConfigurationError(f"provider {self.name!r}: model must be {MODEL_NAME_RULE}")
        if self.max_tokens_field not in MAX_TOKENS_FIELDS:
            raise ConfigurationError(
                f"provider {self.name!r}: max_tokens_field must be one of "
                f"{', '.join(MAX_TOKENS_FIELDS)}"
            )


@dataclass(frozen=True, slots=True)
class TenantSettings:
    tenant_id: str
    tokens_per_minute: int | None = None
    """Per-tenant override of the global limit."""
    model: str | None = None
    """Model sent to providers for this tenant (a per-tenant fine-tune or adapter)."""
    allowed_models: tuple[str, ...] = ()
    """Models the tenant may request; empty means any (the provider pin still applies)."""


@dataclass(frozen=True, slots=True)
class RouterSettings:
    primary: ProviderSettings
    secondary: ProviderSettings
    tenants: Mapping[str, TenantSettings]
    """API key -> tenant."""
    database_path: Path = Path("data/model_router.sqlite3")
    tokens_per_minute: int = 50_000
    window_seconds: int = 60
    default_max_tokens: int = 1024
    """Completion budget reserved, and sent to the provider, when a request specifies none."""
    failover_statuses: tuple[int, ...] = DEFAULT_FAILOVER_STATUSES
    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 30.0
    max_body_bytes: int = 1024 * 1024
    max_response_bytes: int = 8 * 1024 * 1024
    """Largest provider response body the router buffers; bigger ones count as unparseable."""
    eviction_interval_seconds: float = 30.0
    token_estimator: str = "heuristic"
    """``heuristic[:chars_per_token]``, ``tiktoken:<encoding>`` or ``module:factory``."""
    allow_fallback_default: bool = True
    """Whether requests without ``X-Model-Router-Allow-Fallback`` may be served by the secondary."""
    max_cooldown_seconds: float = 60.0
    """Longest pause a provider's ``Retry-After`` header can impose on it."""
    forward_headers: tuple[str, ...] = DEFAULT_FORWARDED_CLIENT_HEADERS
    """Client request headers relayed to providers (lower case); credentials are never eligible."""
    log_level: str = "INFO"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tenants:
            raise ConfigurationError("at least one tenant API key must be configured")
        for key in self.tenants:
            if len(key) < MIN_API_KEY_LENGTH:
                raise ConfigurationError(
                    f"tenant API keys must be at least {MIN_API_KEY_LENGTH} characters"
                )
            if not key.isascii() or not key.isprintable() or " " in key:
                raise ConfigurationError("tenant API keys must be printable ASCII without spaces")
        if self.tokens_per_minute <= 0 or self.window_seconds <= 0:
            raise ConfigurationError("tokens_per_minute and window_seconds must be positive")
        if self.default_max_tokens <= 0:
            raise ConfigurationError("default_max_tokens must be positive")
        if self.primary.name == self.secondary.name:
            raise ConfigurationError("primary and secondary providers must have different names")
        if self.circuit_failure_threshold < 1:
            raise ConfigurationError("circuit_failure_threshold must be at least 1")
        for name in (
            "circuit_recovery_seconds",
            "eviction_interval_seconds",
            "max_cooldown_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive finite number")
        if self.max_body_bytes <= 0:
            raise ConfigurationError("max_body_bytes must be positive")
        if self.max_response_bytes <= 0:
            raise ConfigurationError("max_response_bytes must be positive")
        if self.log_level.upper() not in LOG_LEVELS:
            raise ConfigurationError(f"log_level must be one of {', '.join(LOG_LEVELS)}")
        if not self.token_estimator.strip():
            raise ConfigurationError("token_estimator must not be empty")
        for header in self.forward_headers:
            if header.lower() in _NEVER_FORWARDED or not header.strip():
                raise ConfigurationError(f"forward_headers must not include {header!r}")
        limits: dict[str, tuple[int | None, str | None]] = {}
        for tenant in self.tenants.values():
            if not _TENANT_ID_RE.fullmatch(tenant.tenant_id):
                raise ConfigurationError(
                    "tenant_id must be 1 to 128 printable ASCII characters without "
                    "leading or trailing spaces"
                )
            facts = (tenant.tokens_per_minute, tenant.model)
            if tenant.tenant_id in limits and limits[tenant.tenant_id] != facts:
                raise ConfigurationError(
                    f"tenant {tenant.tenant_id}: its API keys disagree on "
                    "tokens_per_minute or model"
                )
            limits[tenant.tenant_id] = facts
            if tenant.model is not None and not valid_model_name(tenant.model):
                raise ConfigurationError(
                    f"tenant {tenant.tenant_id}: model must be {MODEL_NAME_RULE}"
                )
            if not all(valid_model_name(m) for m in tenant.allowed_models):
                raise ConfigurationError(
                    f"tenant {tenant.tenant_id}: allowed_models entries must be {MODEL_NAME_RULE}"
                )
            if tenant.model and tenant.allowed_models and tenant.model not in tenant.allowed_models:
                raise ConfigurationError(
                    f"tenant {tenant.tenant_id}: its pinned model is not in allowed_models"
                )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RouterSettings:
        source = os.environ if env is None else env

        def get(name: str, default: str | None = None) -> str | None:
            value = source.get(ENV_PREFIX + name)
            return value if value not in (None, "") else default

        def provider(role: str, default_name: str) -> ProviderSettings:
            base_url = get(f"{role}_URL")
            if base_url is None:
                raise ConfigurationError(f"{ENV_PREFIX}{role}_URL is required")
            return ProviderSettings(
                name=get(f"{role}_NAME", default_name) or default_name,
                base_url=base_url.rstrip("/"),
                api_key=get(f"{role}_API_KEY"),
                model=get(f"{role}_MODEL"),
                timeout_ms=int(get(f"{role}_TIMEOUT_MS", "3000") or "3000"),
                dialect=(get(f"{role}_DIALECT", "openai") or "openai").lower(),
                max_concurrency=int(get(f"{role}_MAX_CONCURRENCY", "0") or "0"),
                max_tokens_field=get(f"{role}_MAX_TOKENS_FIELD", "max_tokens") or "max_tokens",
            )

        try:
            allow_development = parse_bool(
                get("ALLOW_DEVELOPMENT_CREDENTIALS", "false") or "false",
                name=f"{ENV_PREFIX}ALLOW_DEVELOPMENT_CREDENTIALS",
            )
            tenants = _load_tenants(
                get("TENANTS_JSON"), get("TENANTS_FILE"), allow_development=allow_development
            )
            statuses = get("FAILOVER_STATUSES")
            return cls(
                primary=provider("PRIMARY", "primary"),
                secondary=provider("SECONDARY", "secondary"),
                tenants=tenants,
                database_path=Path(get("DATABASE_PATH", "data/model_router.sqlite3") or ""),
                tokens_per_minute=int(get("TOKENS_PER_MINUTE", "50000") or "50000"),
                window_seconds=int(get("WINDOW_SECONDS", "60") or "60"),
                default_max_tokens=int(get("DEFAULT_MAX_TOKENS", "1024") or "1024"),
                failover_statuses=tuple(int(s) for s in statuses.split(","))
                if statuses
                else DEFAULT_FAILOVER_STATUSES,
                circuit_failure_threshold=int(get("CIRCUIT_FAILURE_THRESHOLD", "5") or "5"),
                circuit_recovery_seconds=float(get("CIRCUIT_RECOVERY_SECONDS", "30") or "30"),
                max_body_bytes=int(get("MAX_BODY_BYTES", str(1024 * 1024)) or "0"),
                max_response_bytes=int(get("MAX_RESPONSE_BYTES", str(8 * 1024 * 1024)) or "0"),
                eviction_interval_seconds=float(get("EVICTION_INTERVAL_SECONDS", "30") or "30"),
                token_estimator=get("TOKEN_ESTIMATOR", "heuristic") or "heuristic",
                allow_fallback_default=parse_bool(
                    get("ALLOW_FALLBACK_DEFAULT", "true") or "true",
                    name=f"{ENV_PREFIX}ALLOW_FALLBACK_DEFAULT",
                ),
                max_cooldown_seconds=float(get("MAX_COOLDOWN_SECONDS", "60") or "60"),
                forward_headers=tuple(
                    h.strip().lower()
                    for h in (get("FORWARD_HEADERS", "traceparent,tracestate") or "").split(",")
                    if h.strip()
                ),
                log_level=(get("LOG_LEVEL", "INFO") or "INFO").upper(),
            )
        except ValueError as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(f"invalid setting: {exc}") from exc


def _load_tenants(
    inline: str | None, file_path: str | None, *, allow_development: bool = False
) -> dict[str, TenantSettings]:
    """Parse the tenant document. Error messages never quote key material."""
    if inline is None and file_path is None:
        raise ConfigurationError(
            f"set {ENV_PREFIX}TENANTS_FILE or {ENV_PREFIX}TENANTS_JSON; the router does not start "
            "without tenant API keys"
        )
    try:
        text = inline if inline is not None else Path(file_path or "").read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"cannot read {ENV_PREFIX}TENANTS_FILE {file_path!r}: {exc}"
        ) from exc
    try:
        document = json.loads(text, object_pairs_hook=_pairs_without_duplicates)
    except ValueError as exc:
        raise ConfigurationError(f"tenant document is not valid JSON: {exc}") from exc
    entries = document.get("tenants") if isinstance(document, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ConfigurationError('tenant document must be an object with a non-empty "tenants" map')
    if document.get(DEVELOPMENT_MARKER) is True and not allow_development:
        raise ConfigurationError(
            f"the tenant document is marked {DEVELOPMENT_MARKER}: its keys are published; "
            f"set {ENV_PREFIX}ALLOW_DEVELOPMENT_CREDENTIALS=true to use it outside production"
        )
    tenants: dict[str, TenantSettings] = {}
    for position, (api_key, spec) in enumerate(entries.items(), start=1):
        if not isinstance(spec, dict) or not isinstance(spec.get("tenant_id"), str):
            raise ConfigurationError(f"tenant entry {position} lacks tenant_id")
        tenant_id = spec["tenant_id"]
        limit = spec.get("tokens_per_minute")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ConfigurationError(
                f"tenant {tenant_id}: tokens_per_minute must be a positive int"
            )
        model = spec.get("model")
        if model is not None and not valid_model_name(model):
            raise ConfigurationError(
                f"tenant {tenant_id}: model must be a non-empty string ({MODEL_NAME_RULE})"
            )
        allowed = spec.get("allowed_models", [])
        if not isinstance(allowed, list) or not all(valid_model_name(m) for m in allowed):
            raise ConfigurationError(
                f"tenant {tenant_id}: allowed_models must be a list of non-empty strings "
                f"({MODEL_NAME_RULE})"
            )
        tenants[api_key] = TenantSettings(
            tenant_id=tenant_id,
            tokens_per_minute=limit,
            model=model,
            allowed_models=tuple(allowed),
        )
    return tenants


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that refuses duplicate keys instead of silently keeping the last.

    Two entries for one API key would otherwise make the tenant that wins
    depend on file order, which is exactly the kind of surprise a credential
    map must not have. The offending key is identified by position, never
    quoted: it may be a secret.
    """
    document: dict[str, Any] = {}
    for position, (key, value) in enumerate(pairs, start=1):
        if key in document:
            raise ValueError(f"duplicate member: entry {position} of an object repeats a name")
        document[key] = value
    return document
