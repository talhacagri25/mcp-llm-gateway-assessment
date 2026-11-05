"""Streaming behaviour over real sockets: no buffering, bounded first-token latency.

The mock provider paces its deltas; the gateway must relay each one as it
arrives rather than waiting for the end of the response.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import httpx
import pytest

from llm_guardrail_gateway.app import create_app
from llm_guardrail_gateway.config import GuardrailSettings
from llm_guardrail_gateway.redaction.streaming import redact_text
from llm_guardrail_gateway.testing import UvicornTestServer
from mock_llm_provider.app import CANNED_TEXT
from mock_llm_provider.app import create_app as create_provider_app

pytestmark = pytest.mark.integration

CHUNK_SIZE = 6
DELAY_MS = 40
PIECES = -(-len(CANNED_TEXT) // CHUNK_SIZE)


@pytest.fixture(scope="module")
def stack() -> Iterator[tuple[str, str]]:
    with UvicornTestServer(create_provider_app()) as provider:
        settings = GuardrailSettings(
            upstream_base_url=provider.base_url, upstream_api_key="k", allow_unauthenticated=True
        )
        with UvicornTestServer(create_app(settings)) as gateway:
            yield provider.base_url, gateway.base_url


def _timed_events(base_url: str) -> tuple[list[float], str]:
    body = {
        "model": "m",
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
        "mock": {"chunk_size": CHUNK_SIZE, "delay_ms": DELAY_MS},
    }
    arrivals: list[float] = []
    pieces: list[str] = []
    started = time.perf_counter()
    with (
        httpx.Client(timeout=60) as client,
        client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as response,
    ):
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            payload = json.loads(line[5:])
            content = payload["choices"][0]["delta"].get("content")
            if content:
                arrivals.append(time.perf_counter() - started)
                pieces.append(content)
    return arrivals, "".join(pieces)


def test_tokens_are_relayed_incrementally(stack: tuple[str, str]) -> None:
    _, gateway_url = stack
    arrivals, text = _timed_events(gateway_url)
    assert text == redact_text(CANNED_TEXT)
    total = arrivals[-1]
    expected_total = PIECES * DELAY_MS / 1000
    assert total >= 0.5 * expected_total  # the provider really paced the stream
    # First token well before the end: no buffering of the response.
    assert arrivals[0] < 0.25 * total
    # Deliveries are spread over the whole stream rather than bunched at the end.
    midpoint = sum(1 for t in arrivals if t < total / 2)
    assert midpoint >= len(arrivals) * 0.3


def test_first_token_latency_overhead_is_small(stack: tuple[str, str]) -> None:
    provider_url, gateway_url = stack
    direct = min(_timed_events(provider_url)[0][0] for _ in range(3))
    through_gateway = min(_timed_events(gateway_url)[0][0] for _ in range(3))
    # The redactor holds back at most the trailing word of a delta, so the first
    # visible token can arrive one delta later than directly from the provider.
    # Allow that delta plus generous scheduling slack.
    assert through_gateway < direct + DELAY_MS / 1000 + 0.15
