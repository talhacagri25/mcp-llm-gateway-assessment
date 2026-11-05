"""Fixtures: an in-process guardrail gateway in front of the mock provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette

from llm_guardrail_gateway.app import create_app
from llm_guardrail_gateway.config import GuardrailSettings
from llm_guardrail_gateway.sse import SseEvent, SseParser
from mock_llm_provider.app import MockProvider
from mock_llm_provider.app import create_app as create_provider_app

UPSTREAM_URL = "http://upstream"
UPSTREAM_KEY = "sk-upstream-secret"


def _make_settings(**overrides: Any) -> GuardrailSettings:
    base: dict[str, Any] = {
        "upstream_base_url": UPSTREAM_URL,
        "upstream_api_key": UPSTREAM_KEY,
        "allow_unauthenticated": True,  # the open mode is deliberate in tests
    }
    base.update(overrides)
    return GuardrailSettings(**base)


def _events(body: bytes) -> list[SseEvent]:
    parser = SseParser()
    return parser.feed(body) + parser.flush()


def _openai_text(body: bytes) -> str:
    pieces: list[str] = []
    for event in _events(body):
        if not event.has_data or event.data == "[DONE]":
            continue
        for choice in json.loads(event.data).get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content:
                pieces.append(content)
    return "".join(pieces)


def _openai_tool_arguments(body: bytes) -> str:
    pieces: list[str] = []
    for event in _events(body):
        if not event.has_data or event.data == "[DONE]":
            continue
        for choice in json.loads(event.data).get("choices", []):
            for call in choice.get("delta", {}).get("tool_calls", []) or []:
                arguments = call.get("function", {}).get("arguments")
                if arguments:
                    pieces.append(arguments)
    return "".join(pieces)


def _anthropic_text(body: bytes) -> str:
    pieces: list[str] = []
    for event in _events(body):
        if not event.has_data:
            continue
        payload = json.loads(event.data)
        if payload.get("type") == "content_block_delta":
            delta = payload["delta"]
            pieces.append(delta.get("text") or delta.get("partial_json") or "")
    return "".join(pieces)


@pytest.fixture
def make_settings() -> Callable[..., GuardrailSettings]:
    return _make_settings


@pytest.fixture
def sse_events() -> Callable[[bytes], list[SseEvent]]:
    return _events


@pytest.fixture
def openai_text() -> Callable[[bytes], str]:
    return _openai_text


@pytest.fixture
def openai_tool_arguments() -> Callable[[bytes], str]:
    return _openai_tool_arguments


@pytest.fixture
def anthropic_text() -> Callable[[bytes], str]:
    return _anthropic_text


@pytest.fixture
def provider_app() -> Starlette:
    return create_provider_app()


@pytest.fixture
def provider(provider_app: Starlette) -> MockProvider:
    instance: MockProvider = provider_app.state.provider
    return instance


@pytest.fixture
def gateway_for(provider_app: Starlette) -> Callable[..., Any]:
    @asynccontextmanager
    async def factory(
        settings: GuardrailSettings, http_client: httpx.AsyncClient | None = None
    ) -> AsyncIterator[httpx.AsyncClient]:
        upstream = http_client or httpx.AsyncClient(
            transport=httpx.ASGITransport(app=provider_app), base_url=UPSTREAM_URL
        )
        app = create_app(settings, http_client=upstream)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway", timeout=30
            ) as client:
                yield client
        finally:
            await upstream.aclose()

    return factory


@pytest.fixture
async def gateway(gateway_for: Callable[..., Any]) -> AsyncIterator[httpx.AsyncClient]:
    async with gateway_for(_make_settings()) as client:
        yield client
