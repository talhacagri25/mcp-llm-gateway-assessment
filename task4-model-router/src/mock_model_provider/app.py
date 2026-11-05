"""A mock chat-completions provider whose behaviour can be scripted.

Default behaviour comes from the environment (``MOCK_PROVIDER_NAME``,
``MOCK_PROVIDER_BEHAVIOR``, ``MOCK_PROVIDER_DELAY_MS``) and can be overridden
per request through a ``mock`` object in the JSON body, either globally or
keyed by provider name so one request can script the primary and the
secondary differently::

    {"mock": {"primary": {"behavior": "slow", "delay_ms": 4000}, "secondary": {"behavior": "ok"}}}

Behaviours: ``ok``, ``rate_limited`` (429), ``slow`` (sleep then ok),
``error`` (500), ``bad_request`` (400), ``unauthorized`` (401), ``garbage``
(200 that is not JSON), ``flaky`` (429 on odd requests, ok on even ones),
``request_timeout`` (408), ``bad_encoding`` (200 whose body does not match its
``Content-Encoding``), ``nan`` (200 whose JSON carries ``NaN``), ``text_json``
(200 with ``Content-Type: text/json``), ``leaky_bad_request`` (400 whose
message contains internal URLs, paths and a key) and ``no_usage`` (200 without
a usage object, like some self-hosted servers).

``POST /v1/messages`` serves the same behaviours in the Anthropic Messages
shape (plus ``overloaded`` for a 529 and ``tool_use`` for a tool call answer),
so a router configured with the ``anthropic`` dialect can be exercised end to
end.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route


@dataclass
class ProviderState:
    name: str
    default_behavior: str
    default_delay_ms: int
    requests: int = 0
    served: int = 0
    last_payload: dict[str, Any] = field(default_factory=dict)
    last_headers: dict[str, str] = field(default_factory=dict)
    last_path: str = ""


def _error(status: int, message: str, error_type: str) -> Response:
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "code": None}}, status_code=status
    )


def _anthropic_error(status: int, message: str, error_type: str) -> Response:
    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}}, status_code=status
    )


class MockProvider:
    def __init__(self, name: str, behavior: str, delay_ms: int) -> None:
        self.state = ProviderState(name=name, default_behavior=behavior, default_delay_ms=delay_ms)

    def _controls(
        self, payload: dict[str, Any], headers: Mapping[str, str] | None = None
    ) -> tuple[str, int, int]:
        """Behaviour from the body's ``mock`` object, or from an ``X-Mock-Controls`` header.

        The header form exists because a translating router (Anthropic dialect)
        does not forward unknown body keys; tests set it through the router's
        header allow-list.
        """
        raw = payload.get("mock")
        if raw is None and headers is not None and headers.get("x-mock-controls"):
            try:
                raw = json.loads(headers["x-mock-controls"])
            except ValueError:
                raw = None
        options: dict[str, Any] = raw if isinstance(raw, dict) else {}
        scoped = options.get(self.state.name)
        if isinstance(scoped, dict):
            options = {**options, **scoped}
        behavior = str(options.get("behavior", self.state.default_behavior))
        delay_ms = int(options.get("delay_ms", self.state.default_delay_ms))
        completion_tokens = int(options.get("completion_tokens", 20))
        return behavior, delay_ms, completion_tokens

    async def chat_completions(self, request: Request) -> Response:
        self.state.requests += 1
        self.state.last_headers = dict(request.headers)
        self.state.last_path = request.url.path
        try:
            payload = json.loads(await request.body())
        except ValueError:
            return _error(400, "invalid JSON", "invalid_request_error")
        if not isinstance(payload, dict):
            return _error(400, "body must be an object", "invalid_request_error")
        self.state.last_payload = payload
        behavior, delay_ms, completion_tokens = self._controls(payload)

        if behavior == "flaky":
            behavior = "rate_limited" if self.state.requests % 2 == 1 else "ok"
        if behavior == "slow":
            await asyncio.sleep(delay_ms / 1000)
            behavior = "ok"
        if behavior == "rate_limited":
            response = _error(429, "Rate limit reached for this model", "rate_limit_error")
            response.headers["Retry-After"] = "1"
            return response
        if behavior == "error":
            return _error(
                500,
                "The server had an error while processing your request\n"
                "Traceback (most recent call last): internal detail",
                "server_error",
            )
        if behavior == "bad_request":
            return _error(400, "max_tokens is too large for this model", "invalid_request_error")
        if behavior == "unauthorized":
            return _error(401, "Incorrect API key provided", "invalid_request_error")
        if behavior == "garbage":
            return PlainTextResponse("<html>upstream proxy error</html>", status_code=200)
        if behavior == "request_timeout":
            return _error(408, "Request timed out", "timeout_error")
        if behavior == "leaky_bad_request":
            return _error(
                400,
                "model 'gpt' unavailable at http://10.0.0.5:9401/v1 "
                "(handler /srv/gateway/app.py:88, key sk-live-ABCDEFGHIJKLMNOPQRSTUVWXYZ)",
                "invalid_request_error",
            )
        if behavior == "nan":
            return Response(
                '{"id":"nan","object":"chat.completion","choices":[],"usage":{"total_tokens":NaN}}',
                media_type="application/json",
            )

        self.state.served += 1
        body = self._completion(payload, completion_tokens)
        if behavior == "no_usage":
            body.pop("usage")
            return JSONResponse(body, headers={"X-Mock-Provider": self.state.name})
        if behavior == "bad_encoding":
            return Response(
                json.dumps(body),
                media_type="application/json",
                headers={"Content-Encoding": "gzip"},  # the body is not actually compressed
            )
        if behavior == "text_json":
            return Response(json.dumps(body), media_type="text/json")
        return JSONResponse(body, headers={"X-Mock-Provider": self.state.name})

    def _completion(self, payload: dict[str, Any], completion_tokens: int) -> dict[str, Any]:
        prompt_source = payload.get("messages", payload.get("prompt", ""))
        prompt_tokens = max(1, math.ceil(len(json.dumps(prompt_source)) / 4))
        model = str(payload.get("model", "mock-model"))
        return {
            "id": f"chatcmpl-{self.state.name}-{self.state.requests}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Hello from {self.state.name} ({model}).",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # -- Anthropic Messages dialect -------------------------------------------

    async def messages(self, request: Request) -> Response:
        self.state.requests += 1
        self.state.last_headers = dict(request.headers)
        self.state.last_path = request.url.path
        try:
            payload = json.loads(await request.body())
        except ValueError:
            return _anthropic_error(400, "invalid JSON", "invalid_request_error")
        if not isinstance(payload, dict):
            return _anthropic_error(400, "body must be an object", "invalid_request_error")
        self.state.last_payload = payload
        behavior, delay_ms, completion_tokens = self._controls(payload, request.headers)
        if behavior == "flaky":
            behavior = "rate_limited" if self.state.requests % 2 == 1 else "ok"
        if behavior == "slow":
            await asyncio.sleep(delay_ms / 1000)
            behavior = "ok"
        if behavior == "rate_limited":
            response = _anthropic_error(429, "Rate limited", "rate_limit_error")
            response.headers["Retry-After"] = "1"
            return response
        if behavior == "overloaded":
            return _anthropic_error(529, "Overloaded", "overloaded_error")
        if behavior == "error":
            return _anthropic_error(500, "Internal server error", "api_error")
        if behavior == "bad_request":
            return _anthropic_error(
                400, "max_tokens: must be greater than 0", "invalid_request_error"
            )
        if behavior == "unauthorized":
            return _anthropic_error(401, "invalid x-api-key", "authentication_error")
        if behavior == "garbage":
            return PlainTextResponse("<html>upstream proxy error</html>", status_code=200)

        self.state.served += 1
        model = str(payload.get("model", "mock-model"))
        prompt_tokens = max(1, math.ceil(len(json.dumps(payload.get("messages", []))) / 4))
        content: list[dict[str, Any]]
        stop_reason = "end_turn"
        if behavior == "tool_use" and payload.get("tools"):
            content = [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_mock_1",
                    "name": payload["tools"][0]["name"],
                    "input": {"city": "Paris"},
                },
            ]
            stop_reason = "tool_use"
        else:
            content = [{"type": "text", "text": f"Hello from {self.state.name} ({model})."}]
        return JSONResponse(
            {
                "id": f"msg_{self.state.name}_{self.state.requests}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
            },
            headers={"X-Mock-Provider": self.state.name},
        )

    async def stats(self, request: Request) -> Response:
        return JSONResponse(
            {
                "name": self.state.name,
                "requests": self.state.requests,
                "served": self.state.served,
                "default_behavior": self.state.default_behavior,
            }
        )

    async def health(self, request: Request) -> Response:
        return JSONResponse(
            {"status": "ok", "service": "mock-model-provider", "name": self.state.name}
        )


def create_app(
    name: str | None = None, behavior: str | None = None, delay_ms: int | None = None
) -> Starlette:
    provider = MockProvider(
        name=name or os.environ.get("MOCK_PROVIDER_NAME", "provider"),
        behavior=behavior or os.environ.get("MOCK_PROVIDER_BEHAVIOR", "ok"),
        delay_ms=(
            delay_ms
            if delay_ms is not None
            else int(os.environ.get("MOCK_PROVIDER_DELAY_MS", "5000"))
        ),
    )
    app = Starlette(
        routes=[
            Route("/v1/chat/completions", provider.chat_completions, methods=["POST"]),
            Route("/v1/messages", provider.messages, methods=["POST"]),
            Route("/__mock/stats", provider.stats, methods=["GET"]),
            Route("/healthz", provider.health, methods=["GET"]),
        ]
    )
    app.state.provider = provider
    return app
