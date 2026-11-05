"""Token estimation, usage parsing, calibration, estimator plugins and the circuit breaker."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from model_router.circuit_breaker import CircuitBreaker, CircuitState
from model_router.token_estimator import (
    Calibration,
    HeuristicTokenEstimator,
    completion_tokens_from_body,
    load_estimator,
    prompt_tokens,
    reported_total_tokens,
    reported_usage,
    requested_choices,
    requested_completion_tokens,
)

HEURISTIC = HeuristicTokenEstimator()


def test_estimate_grows_with_content_and_counts_every_message() -> None:
    short = prompt_tokens({"messages": [{"role": "user", "content": "hi"}]}, HEURISTIC)
    long = prompt_tokens({"messages": [{"role": "user", "content": "hi" * 500}]}, HEURISTIC)
    two = prompt_tokens(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": "hi"}]},
        HEURISTIC,
    )
    assert short < long
    assert two > short
    parts = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x" * 40}]}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    assert prompt_tokens(parts, HEURISTIC) > prompt_tokens({"prompt": "x" * 40}, HEURISTIC)
    odd = {"messages": [{"role": "user", "content": [{"type": "text", "text": None}, 5]}]}
    assert prompt_tokens(odd, HEURISTIC) > 0  # never raises on odd shapes


def test_heuristic_ratio_and_validation() -> None:
    assert HeuristicTokenEstimator(2.0).count("abcd") == 2
    assert HeuristicTokenEstimator().count("") == 0
    with pytest.raises(ValueError, match="positive"):
        HeuristicTokenEstimator(0)


def test_requested_completion_tokens_and_choices() -> None:
    assert requested_completion_tokens({"max_tokens": 10}, 1024) == 10
    assert requested_completion_tokens({"max_completion_tokens": 7, "max_tokens": 10}, 1024) == 7
    assert requested_completion_tokens({}, 1024) == 1024
    assert requested_completion_tokens({"max_tokens": -5}, 1024) == 1024
    assert requested_choices({"n": 3}) == 3
    assert requested_choices({"n": True}) == 1
    assert requested_choices({}) == 1


def test_reported_usage_accepts_both_shapes_and_rejects_nonsense() -> None:
    assert reported_usage({"usage": {"total_tokens": 42}}) == (None, None, 42)
    assert reported_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}) == (10, 5, 15)
    assert reported_usage({"usage": {"input_tokens": 7, "output_tokens": 3}}) == (7, 3, 10)
    assert reported_usage({"usage": {}}) is None
    assert reported_usage({}) is None
    assert reported_usage({"usage": {"total_tokens": "42"}}) is None
    assert reported_usage({"usage": {"total_tokens": -1}}) is None
    assert reported_usage({"usage": {"total_tokens": 10**12}}) is None
    assert reported_usage({"usage": {"prompt_tokens": -5, "completion_tokens": 50}}) is None
    assert reported_usage({"usage": {"total_tokens": True}}) is None
    assert reported_total_tokens({"usage": {"total_tokens": 0}}) == 0


def test_completion_tokens_are_estimated_from_every_text_channel() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": "x" * 40,
                    "reasoning_content": "y" * 40,
                    "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}],
                }
            }
        ]
    }
    assert completion_tokens_from_body(body, HEURISTIC) > 20
    assert completion_tokens_from_body({"choices": "nope"}, HEURISTIC) == 0


class _MemoryStore:
    def __init__(self) -> None:
        self.saved: dict[tuple[str, str], tuple[float, int]] = {}

    def load_calibration(self) -> dict[tuple[str, str], tuple[float, int]]:
        return dict(self.saved)

    def save_calibration(self, tenant_id: str, provider: str, factor: float, samples: int) -> None:
        self.saved[(tenant_id, provider)] = (factor, samples)


def test_calibration_learns_clamps_and_persists() -> None:
    store = _MemoryStore()
    calibration = Calibration(store, alpha=0.5)
    assert calibration.factor("t", "p") == 1.0
    calibration.observe("t", "p", estimated_prompt=100, actual_prompt=150)
    assert calibration.factor("t", "p") == 1.5  # the first sample is taken as is
    calibration.observe("t", "p", estimated_prompt=100, actual_prompt=250)
    assert calibration.factor("t", "p") == pytest.approx(2.0)  # moving average
    calibration.observe("t", "p", estimated_prompt=100, actual_prompt=100_000)
    assert calibration.factor("t", "p") <= 4.0  # clamped
    calibration.observe("t", "p", estimated_prompt=5, actual_prompt=500)  # too small to learn
    assert calibration.samples("t", "p") == 3
    assert calibration.worst_factor("t", ["p", "q"]) == calibration.factor("t", "p")
    assert store.saved[("t", "p")][1] == 3
    reloaded = Calibration(store)
    assert reloaded.factor("t", "p") == calibration.factor("t", "p")


def test_calibration_is_isolated_per_tenant() -> None:
    calibration = Calibration(_MemoryStore())
    calibration.observe("loud", "p", estimated_prompt=100, actual_prompt=400)
    assert calibration.factor("loud", "p") == 4.0
    assert calibration.factor("quiet", "p") == 1.0  # one tenant cannot inflate another's cost
    assert calibration.worst_factor("quiet", ["p", "q"]) == 1.0
    assert calibration.factors_for("loud", ["p", "q"]) == {"p": 4.0, "q": 1.0}


def test_persisted_factors_are_clamped_on_load() -> None:
    store = _MemoryStore()
    store.saved[("t", "p")] = (99.0, -3)  # a tampered or corrupt row
    store.saved[("t", "q")] = (float("nan"), 2)
    calibration = Calibration(store)
    assert calibration.factor("t", "p") == 4.0
    assert calibration.samples("t", "p") == 0
    assert calibration.factor("t", "q") == 1.0


def test_calibration_survives_a_broken_store() -> None:
    class Broken:
        def load_calibration(self) -> dict[tuple[str, str], tuple[float, int]]:
            raise OSError("disk gone")

        def save_calibration(
            self, tenant_id: str, provider: str, factor: float, samples: int
        ) -> None:
            raise OSError("disk gone")

    calibration = Calibration(Broken())
    calibration.observe("t", "p", 100, 200)
    assert calibration.factor("t", "p") == 2.0


def test_media_parts_and_non_ascii_text_are_priced_conservatively() -> None:
    text_only = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    with_image = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
    }
    assert prompt_tokens(with_image, HEURISTIC) - prompt_tokens(text_only, HEURISTIC) >= 1000
    ascii_text = "a" * 400
    cjk_text = "\u4e2d" * 400  # same character count, roughly one token per character
    assert HEURISTIC.count(cjk_text) > 2 * HEURISTIC.count(ascii_text)


def test_reported_total_below_its_parts_is_corrected_upwards() -> None:
    usage = reported_usage(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 1}}
    )
    assert usage is not None
    assert usage.total == 150


def test_estimator_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_estimator("heuristic").count("abcdefgh") == 2
    assert load_estimator("heuristic:2").count("abcd") == 2
    plugin = tmp_path / "my_tokenizer.py"
    plugin.write_text(
        "class Counter:\n"
        "    def count(self, text):\n"
        "        return len(text.split())\n"
        "def make():\n"
        "    return Counter()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("my_tokenizer", None)
    assert load_estimator("my_tokenizer:make").count("one two three") == 3
    for bad in ("nonsense", "tiktoken:", "my_tokenizer:missing", "no.such.module:make"):
        with pytest.raises(ValueError, match="estimator"):
            load_estimator(bad)
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        with pytest.raises(ValueError, match="tiktoken"):
            load_estimator("tiktoken:o200k_base")
    else:  # pragma: no cover - depends on the environment
        assert load_estimator("tiktoken:o200k_base").count("hello world") >= 2


def test_circuit_breaker_lifecycle() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=10, clock=lambda: now[0])
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.allows_request()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allows_request()
    now[0] = 10
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allows_request()  # one trial
    assert not breaker.allows_request()  # second caller waits
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allows_request()


def test_circuit_breaker_cooldown_from_retry_after() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=5, recovery_seconds=30, clock=lambda: now[0])
    breaker.cool_down(5)
    assert breaker.cooling_down
    assert not breaker.allows_request()
    breaker.cool_down(2)  # a shorter hint never shortens an existing cooldown
    now[0] = 4.9
    assert not breaker.allows_request()
    now[0] = 5
    assert breaker.allows_request()
    breaker.cool_down(0)
    assert breaker.allows_request()
    breaker.cool_down(10)
    breaker.record_success()  # a healthy answer lifts the pause
    assert breaker.allows_request()


def test_as_dict_shapes_are_json_friendly() -> None:
    from model_router.router import Attempt

    data: dict[str, Any] = Attempt("p", "success", 200, 1.5, "m").as_dict()
    assert data == {
        "provider": "p",
        "outcome": "success",
        "status": 200,
        "latency_ms": 1.5,
        "model": "m",
    }
