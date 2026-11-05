"""Token estimation, pluggable tokenizers and online calibration.

Admission needs a token count before the provider has seen the request. The
default is a characters-per-token heuristic that errs on the high side, and
deployments can plug in a real tokenizer: ``tiktoken:<encoding>`` when the
optional ``tiktoken`` package is installed, or any ``module:factory`` whose
object has a ``count(text) -> int`` method (a HuggingFace tokenizer wrapper
for a fine-tuned open-weight model, for example).

Whatever the estimator, a fine-tuned model's real tokenizer will disagree
with it by some factor. ``Calibration`` learns that factor per tenant and
provider from the usage the provider reports (an exponential moving average
of reported over estimated prompt tokens, clamped to a sane range and
persisted in the ledger database). Factors are per tenant on purpose: one
tenant's prompts must never change another tenant's admissions. The router
reserves with the worst factor among its providers, so the estimate converges
on the truth without any tokenizer knowledge at all.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, Protocol

logger = logging.getLogger(__name__)

DEFAULT_CHARS_PER_TOKEN = 4.0
PER_MESSAGE_OVERHEAD = 4
REPLY_PRIMING = 3
IMAGE_PART_TOKENS = 1600
"""Reserved for an image part: the upper end of what providers bill per image."""
MEDIA_PART_TYPES = frozenset({"image_url", "image", "input_image", "input_audio", "audio", "file"})
MAX_REPORTED_TOKENS = 10_000_000
"""Usage figures beyond this are treated as provider bugs and ignored."""
FACTOR_FLOOR = 0.5
FACTOR_CEILING = 4.0


class TokenEstimator(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenEstimator:
    """About one token per ``chars_per_token`` ASCII characters, more for other scripts.

    Characters outside ASCII (CJK, emoji, accented letters) cost roughly one
    token per code point with common tokenizers, so they are counted by their
    extra UTF-8 bytes rather than by the ASCII ratio.
    """

    def __init__(self, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> None:
        if not math.isfinite(chars_per_token) or chars_per_token <= 0:
            raise ValueError("chars_per_token must be a positive number")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        if text.isascii():
            return math.ceil(len(text) / self.chars_per_token)
        extra_bytes = len(text.encode("utf-8")) - len(text)
        return math.ceil(len(text) / self.chars_per_token + extra_bytes / 2)


class TiktokenEstimator:
    """Exact counts for OpenAI tokenisers; needs the optional ``tiktoken`` package."""

    def __init__(self, encoding_name: str) -> None:
        try:
            tiktoken = importlib.import_module("tiktoken")
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ValueError("tiktoken is not installed") from exc
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=())) if text else 0


def load_estimator(spec: str) -> TokenEstimator:
    """Build the estimator named by ``spec``.

    * ``heuristic`` or ``heuristic:3.5`` (characters per token)
    * ``tiktoken:o200k_base`` (any tiktoken encoding name)
    * ``package.module:factory`` for a custom tokenizer; the factory takes no
      arguments and returns an object with ``count(text) -> int``

    The last form imports and runs code named by configuration; it is
    configuration with the authority of code, like the rest of the settings.
    """
    kind, _, argument = spec.partition(":")
    if kind == "heuristic":
        ratio = float(argument) if argument else DEFAULT_CHARS_PER_TOKEN
        if not 0.25 <= ratio <= 64:
            raise ValueError("heuristic chars per token must be between 0.25 and 64")
        return HeuristicTokenEstimator(ratio)
    if kind == "tiktoken":
        if not argument:
            raise ValueError("tiktoken estimator needs an encoding name, e.g. tiktoken:o200k_base")
        return TiktokenEstimator(argument)
    if kind and argument:
        try:
            module = importlib.import_module(kind)
            factory = getattr(module, argument)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot load token estimator {spec!r}: {exc}") from exc
        estimator = factory()
        if not callable(getattr(estimator, "count", None)):
            raise ValueError(f"token estimator {spec!r} has no count() method")
        probe = estimator.count("probe text")
        if isinstance(probe, bool) or not isinstance(probe, int) or probe < 1:
            raise ValueError(f"token estimator {spec!r} must count a short text as >= 1 tokens")
        return estimator  # type: ignore[no-any-return]
    raise ValueError(f"unknown token estimator {spec!r}")


# --- counting ---------------------------------------------------------------


def _content_tokens(content: Any, estimator: TokenEstimator) -> int:
    """Tokens of a message body in any of the common shapes; never raises on odd input."""
    if isinstance(content, str):
        return estimator.count(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in MEDIA_PART_TYPES:
                    total += IMAGE_PART_TOKENS
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    total += estimator.count(text)
                else:
                    total += estimator.count(json.dumps(part, default=str))
            else:
                total += estimator.count(str(part))
        return total
    return estimator.count(json.dumps(content, default=str)) if content is not None else 0


def prompt_tokens(payload: dict[str, Any], estimator: TokenEstimator) -> int:
    """Tokens the prompt side of a request will cost, before any calibration."""
    total = 0
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                total += PER_MESSAGE_OVERHEAD + _content_tokens(message.get("content"), estimator)
                if message.get("tool_calls"):
                    total += estimator.count(json.dumps(message["tool_calls"], default=str))
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        total += estimator.count(prompt)
    system = payload.get("system")
    if isinstance(system, str):
        total += estimator.count(system)
    tools = payload.get("tools")
    if tools:
        total += estimator.count(json.dumps(tools, default=str))
    return total + REPLY_PRIMING


def completion_tokens_from_body(body: Any, estimator: TokenEstimator) -> int:
    """Estimate of the generated tokens in a Chat Completions response (no usage object)."""
    if not isinstance(body, dict) or not isinstance(body.get("choices"), list):
        return 0
    total = 0
    for choice in body["choices"]:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        total += _content_tokens(message.get("content"), estimator)
        for key in ("reasoning_content", "refusal"):
            if isinstance(message.get(key), str):
                total += estimator.count(message[key])
        if isinstance(message.get("tool_calls"), list):
            total += estimator.count(json.dumps(message["tool_calls"], default=str))
    return total


def requested_choices(payload: dict[str, Any]) -> int:
    """How many completions the caller asked for (``n``); each one consumes the budget."""
    value = payload.get("n")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 1


def requested_completion_tokens(payload: dict[str, Any], default: int) -> int:
    for key in ("max_completion_tokens", "max_tokens"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


class Usage(NamedTuple):
    prompt: int | None
    completion: int | None
    total: int


def _plausible(value: int) -> int | None:
    return value if 0 <= value <= MAX_REPORTED_TOKENS else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return _plausible(value)


def reported_usage(body: Any) -> Usage | None:
    """Usage from a provider response in the OpenAI or Anthropic shape, or ``None``.

    Implausible figures (negative, absurdly large) disqualify the whole object,
    in which case the caller falls back to estimates. A ``total_tokens`` below
    the sum of its parts is corrected upwards: the parts are what was consumed.
    """
    if not isinstance(body, dict) or not isinstance(body.get("usage"), dict):
        return None
    usage = body["usage"]
    prompt = _int_or_none(usage.get("prompt_tokens", usage.get("input_tokens")))
    completion = _int_or_none(usage.get("completion_tokens", usage.get("output_tokens")))
    total = _int_or_none(usage.get("total_tokens"))
    if prompt is not None and completion is not None:
        parts = _plausible(prompt + completion)
        total = parts if total is None or (parts is not None and parts > total) else total
    if total is None:
        return None
    return Usage(prompt, completion, total)


def reported_total_tokens(body: dict[str, Any]) -> int | None:
    usage = reported_usage(body)
    return usage.total if usage else None


# --- calibration ------------------------------------------------------------

CalibrationKey = tuple[str, str]
"""``(tenant_id, provider_name)``."""


class CalibrationStore(Protocol):
    def load_calibration(self) -> dict[CalibrationKey, tuple[float, int]]: ...

    def save_calibration(
        self, tenant_id: str, provider: str, factor: float, samples: int
    ) -> None: ...


def clamp_factor(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return min(FACTOR_CEILING, max(FACTOR_FLOOR, value))


class Calibration:
    """Per tenant and provider ratio of reported to estimated prompt tokens, learned online."""

    def __init__(
        self,
        store: CalibrationStore | None = None,
        *,
        alpha: float = 0.2,
        min_estimate: int = 20,
    ) -> None:
        self._store = store
        self._alpha = alpha
        self._min_estimate = min_estimate
        self._factors: dict[CalibrationKey, tuple[float, int]] = {}
        if store is not None:
            try:
                loaded = store.load_calibration()
            except Exception:
                logger.exception(
                    "calibration could not be loaded; starting from 1.0",
                    extra={"event": "calibration.load_failed"},
                )
            else:
                # Persisted values are data, not trusted code: clamp them like observations.
                self._factors = {
                    key: (clamp_factor(factor), max(0, samples))
                    for key, (factor, samples) in loaded.items()
                }

    def factor(self, tenant_id: str, provider: str) -> float:
        return self._factors.get((tenant_id, provider), (1.0, 0))[0]

    def samples(self, tenant_id: str, provider: str) -> int:
        return self._factors.get((tenant_id, provider), (1.0, 0))[1]

    def factors_for(self, tenant_id: str, providers: Iterable[str]) -> dict[str, float]:
        return {provider: round(self.factor(tenant_id, provider), 3) for provider in providers}

    def worst_factor(self, tenant_id: str, providers: Iterable[str]) -> float:
        """The largest factor among ``providers`` for this tenant: reservations cover the worst."""
        return max((self.factor(tenant_id, p) for p in providers), default=1.0)

    def observe(
        self, tenant_id: str, provider: str, estimated_prompt: int, actual_prompt: int
    ) -> None:
        """Fold one observation in; tiny prompts are skipped because overheads dominate them."""
        if estimated_prompt < self._min_estimate or actual_prompt <= 0:
            return
        ratio = clamp_factor(actual_prompt / estimated_prompt)
        key = (tenant_id, provider)
        previous, samples = self._factors.get(key, (1.0, 0))
        updated = ratio if samples == 0 else (1 - self._alpha) * previous + self._alpha * ratio
        self._factors[key] = (updated, samples + 1)
        self._persist(tenant_id, provider, updated, samples + 1)

    def _persist(self, tenant_id: str, provider: str, factor: float, samples: int) -> None:
        if self._store is None:
            return
        try:
            self._store.save_calibration(tenant_id, provider, factor, samples)
        except Exception:
            logger.exception(
                "calibration could not be persisted", extra={"event": "calibration.save_failed"}
            )


EstimatorFactory = Callable[[], TokenEstimator]
