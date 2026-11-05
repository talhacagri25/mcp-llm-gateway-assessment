"""End-to-end behaviour of the guardrail gateway with the mock provider (in-process)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from llm_guardrail_gateway.adapters import OpenAIChatAdapter
from llm_guardrail_gateway.config import GuardrailSettings
from llm_guardrail_gateway.redaction.streaming import redact_text
from llm_guardrail_gateway.sse import SseEvent
from mock_llm_provider.app import CANNED_TEXT, TOOL_ARGUMENTS, MockProvider

Settings = Callable[..., GuardrailSettings]
Factory = Callable[..., Any]
Text = Callable[[bytes], str]
Events = Callable[[bytes], list[SseEvent]]

RAW_PII = ("maria.santos@example.com", "536-22-8041", "4111 1111 1111 1111")


def chat(stream: bool = True, text: str | None = None, **mock: Any) -> dict[str, Any]:
    content = f"echo:{text}" if text is not None else "hello"
    body: dict[str, Any] = {
        "model": "mock-gpt",
        "stream": stream,
        "messages": [{"role": "user", "content": content}],
    }
    if mock:
        body["mock"] = mock
    return body


def messages(stream: bool = True, text: str | None = None, **mock: Any) -> dict[str, Any]:
    body = chat(stream, text, **mock)
    body["max_tokens"] = 256
    body["model"] = "mock-claude"
    return body


# ---------------------------------------------------------------------------
# Streaming redaction, OpenAI dialect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13, 64, 1000])
async def test_streaming_redaction_is_exact_for_every_chunk_size(
    gateway: httpx.AsyncClient, openai_text: Text, sse_events: Events, size: int
) -> None:
    response = await gateway.post("/v1/chat/completions", json=chat(chunk_size=size))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["x-request-id"]
    text = openai_text(response.content)
    assert text == redact_text(CANNED_TEXT)
    assert not any(pii in response.text for pii in RAW_PII)
    events = sse_events(response.content)
    assert events[-1].data == "[DONE]"
    # one role chunk + content chunks + one finish chunk + [DONE]: nothing added or dropped
    assert len(events) == 1 + math.ceil(len(CANNED_TEXT) / size) + 1 + 1


async def test_streaming_keeps_event_structure_and_usage(
    gateway: httpx.AsyncClient, sse_events: Events
) -> None:
    body = chat(chunk_size=6)
    body["stream_options"] = {"include_usage": True}
    response = await gateway.post("/v1/chat/completions", json=body)
    events = [json.loads(e.data) for e in sse_events(response.content) if e.data != "[DONE]"]
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    assert all(e["id"] == events[0]["id"] for e in events)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    finish = [e for e in events if e["choices"] and e["choices"][0]["finish_reason"]]
    assert len(finish) == 1
    assert events[-1]["usage"]["total_tokens"] > 0


async def test_streaming_tool_call_arguments_are_redacted(
    gateway: httpx.AsyncClient, openai_tool_arguments: Text
) -> None:
    response = await gateway.post("/v1/chat/completions", json=chat(mode="tool_call", chunk_size=3))
    arguments = openai_tool_arguments(response.content)
    assert json.loads(arguments) == {
        "recipient": "[REDACTED]",
        "ssn": "[REDACTED]",
        "note": "call 555-0100",
    }


async def test_custom_text_with_pii_on_boundaries(
    gateway: httpx.AsyncClient, openai_text: Text
) -> None:
    text = "Card 4111-1111-1111-1111, mail a.b@c.io, SSN 536 22 8041 end"
    response = await gateway.post("/v1/chat/completions", json=chat(text=text, chunk_size=2))
    assert openai_text(response.content) == "Card [REDACTED], mail [REDACTED], SSN [REDACTED] end"


async def test_aborted_upstream_stream_flushes_held_text(
    gateway: httpx.AsyncClient, openai_text: Text, sse_events: Events
) -> None:
    size = 5
    response = await gateway.post("/v1/chat/completions", json=chat(mode="abort", chunk_size=size))
    pieces = math.ceil(len(CANNED_TEXT) / size)
    sent = CANNED_TEXT[: (pieces // 2) * size]
    assert openai_text(response.content) == redact_text(sent)
    assert sse_events(response.content)[-1].data != "[DONE]"


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


async def test_non_streaming_response_is_redacted(gateway: httpx.AsyncClient) -> None:
    response = await gateway.post("/v1/chat/completions", json=chat(stream=False))
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == redact_text(CANNED_TEXT)
    assert body["usage"]["total_tokens"] > 0


async def test_non_streaming_tool_call_is_redacted(gateway: httpx.AsyncClient) -> None:
    response = await gateway.post("/v1/chat/completions", json=chat(stream=False, mode="tool_call"))
    arguments = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == json.loads(redact_text(TOOL_ARGUMENTS))


# ---------------------------------------------------------------------------
# Anthropic dialect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 4, 9, 500])
async def test_anthropic_streaming_redaction(
    gateway: httpx.AsyncClient, anthropic_text: Text, sse_events: Events, size: int
) -> None:
    response = await gateway.post("/v1/messages", json=messages(chunk_size=size))
    assert response.status_code == 200
    assert anthropic_text(response.content) == redact_text(CANNED_TEXT)
    kinds = [e.event for e in sse_events(response.content)]
    assert kinds[:3] == ["message_start", "ping", "content_block_start"]
    assert kinds[-3:] == ["content_block_stop", "message_delta", "message_stop"]


async def test_anthropic_tool_input_streaming_and_non_streaming(
    gateway: httpx.AsyncClient, anthropic_text: Text
) -> None:
    streamed = await gateway.post("/v1/messages", json=messages(mode="tool_call", chunk_size=3))
    assert json.loads(anthropic_text(streamed.content)) == json.loads(redact_text(TOOL_ARGUMENTS))
    buffered = await gateway.post("/v1/messages", json=messages(stream=False, mode="tool_call"))
    assert buffered.json()["content"][0]["input"] == json.loads(redact_text(TOOL_ARGUMENTS))


# ---------------------------------------------------------------------------
# Request validation and authentication
# ---------------------------------------------------------------------------


async def test_gateway_api_keys_are_enforced_when_configured(
    gateway_for: Factory, make_settings: Settings
) -> None:
    async with gateway_for(make_settings(api_keys=("gw-key-1",))) as client:
        missing = await client.post("/v1/chat/completions", json=chat(stream=False))
        wrong = await client.post(
            "/v1/chat/completions",
            json=chat(stream=False),
            headers={"Authorization": "Bearer nope"},
        )
        right = await client.post(
            "/v1/chat/completions",
            json=chat(stream=False),
            headers={"Authorization": "Bearer gw-key-1"},
        )
    assert missing.status_code == 401
    assert missing.json()["error"]["type"] == "authentication_error"
    assert wrong.status_code == 401
    assert right.status_code == 200


@pytest.mark.parametrize(
    ("content", "status", "error_type"),
    [
        (b"{not json", 400, "invalid_request"),
        (b"[1, 2, 3]", 400, "invalid_request"),
        (b"[" * 1000, 400, "invalid_request"),  # deep nesting, within the body limit
        (b"x" * 2048, 413, "request_too_large"),
    ],
)
async def test_invalid_request_bodies(
    gateway_for: Factory, make_settings: Settings, content: bytes, status: int, error_type: str
) -> None:
    async with gateway_for(make_settings(max_body_bytes=1024)) as client:
        response = await client.post(
            "/v1/chat/completions", content=content, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == status
    body = response.json()["error"]
    assert body["type"] == error_type
    assert body["request_id"] == response.headers["x-request-id"]


async def test_health_endpoint_is_liveness_only(gateway: httpx.AsyncClient) -> None:
    response = await gateway.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}  # no detector or dialect inventory for strangers
    assert response.headers["cache-control"] == "no-store"


# ---------------------------------------------------------------------------
# Upstream header handling
# ---------------------------------------------------------------------------


async def test_upstream_credentials_are_injected_and_client_headers_filtered(
    gateway: httpx.AsyncClient, provider: MockProvider
) -> None:
    await gateway.post(
        "/v1/chat/completions",
        json=chat(stream=False),
        headers={
            "Authorization": "Bearer client-secret",
            "X-Mock-Mode": "error",  # not on the allow-list: must not reach the provider
            "traceparent": "00-abc-def-01",
            "X-Request-Id": "trace-9",
        },
    )
    seen = provider.state.last_headers
    assert seen["authorization"] == "Bearer sk-upstream-secret"
    assert "x-mock-mode" not in seen
    assert seen["traceparent"] == "00-abc-def-01"
    assert seen["x-request-id"] == "trace-9"
    assert seen["accept-encoding"] == "identity"
    assert seen["user-agent"].startswith("llm-guardrail-gateway/")

    await gateway.post("/v1/messages", json=messages(stream=False))
    seen = provider.state.last_headers
    assert seen["x-api-key"] == "sk-upstream-secret"
    assert "authorization" not in seen
    assert seen["anthropic-version"] == "2023-06-01"


async def test_bring_your_own_key_mode_passes_the_client_credential(
    gateway_for: Factory, make_settings: Settings, provider: MockProvider
) -> None:
    async with gateway_for(make_settings(upstream_api_key=None)) as client:
        await client.post(
            "/v1/chat/completions",
            json=chat(stream=False),
            headers={"Authorization": "Bearer mine"},
        )
    assert provider.state.last_headers["authorization"] == "Bearer mine"


# ---------------------------------------------------------------------------
# Upstream failures
# ---------------------------------------------------------------------------


async def test_upstream_error_document_is_relayed_without_internals(
    gateway: httpx.AsyncClient,
) -> None:
    response = await gateway.post("/v1/chat/completions", json=chat(mode="error"))
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert "Traceback" not in response.text


@pytest.mark.parametrize(
    ("exc", "status", "error_type"),
    [
        (httpx.ConnectError("refused"), 502, "upstream_unavailable"),
        (httpx.ConnectTimeout("slow"), 504, "upstream_timeout"),
        (httpx.ReadTimeout("slow"), 504, "upstream_timeout"),
    ],
)
async def test_transport_failures_are_standardised(
    gateway_for: Factory, make_settings: Settings, exc: Exception, status: int, error_type: str
) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        raise exc

    client = httpx.AsyncClient(transport=httpx.MockTransport(failing))
    async with gateway_for(make_settings(), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat())
    assert response.status_code == status
    assert response.json()["error"]["type"] == error_type
    assert "refused" not in response.text
    assert "slow" not in response.text


async def test_unparseable_upstream_success_fails_closed(
    gateway_for: Factory, make_settings: Settings
) -> None:
    def plain(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="mail a@b.io", headers={"content-type": "text/plain"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(plain))
    async with gateway_for(make_settings(), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat())
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_unparseable"
    assert "a@b.io" not in response.text


async def test_stream_requested_but_json_returned_is_still_redacted(
    gateway_for: Factory, make_settings: Settings
) -> None:
    def json_only(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ssn 536-22-8041"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(json_only))
    async with gateway_for(make_settings(), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat())
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ssn [REDACTED]"


async def test_mid_stream_upstream_failure_is_reported_in_band(
    gateway_for: Factory, make_settings: Settings, sse_events: Events
) -> None:
    async def broken_stream() -> Any:
        yield (
            b'data: {"choices":[{"index":0,"delta":{"content":"hello "},"finish_reason":null}]}\n\n'
        )
        raise httpx.ReadError("connection reset")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_AsyncByteStream(broken_stream()),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat())
    events = sse_events(response.content)
    assert events[-1].event == "error"
    assert json.loads(events[-1].data)["error"]["type"] == "upstream_stream_error"
    assert "connection reset" not in response.text


class _AsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, iterator: Any) -> None:
        self._iterator = iterator

    async def __aiter__(self) -> Any:
        async for chunk in self._iterator:
            yield chunk


async def test_gateway_defect_mid_stream_fails_closed(
    gateway: httpx.AsyncClient, sse_events: Events, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(self: OpenAIChatAdapter, event: Any) -> Any:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(OpenAIChatAdapter, "process", explode)
    response = await gateway.post("/v1/chat/completions", json=chat())
    assert response.status_code == 200
    events = sse_events(response.content)
    assert [e.event for e in events] == ["error"]
    assert json.loads(events[0].data)["error"]["type"] == "gateway_stream_error"
    assert "secret internal detail" not in response.text
    assert not any(pii in response.text for pii in RAW_PII)


async def test_deeply_nested_upstream_event_cannot_crash_the_stream(
    gateway_for: Factory, make_settings: Settings, sse_events: Events
) -> None:
    bomb = b"data: " + b"[" * 100_000 + b"\n\n"
    good = (
        b'data: {"choices":[{"index":0,"delta":{"content":"ok a@b.io"},'
        b'"finish_reason":"stop"}]}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=bomb + good + b"data: [DONE]\n\n",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat())
    events = sse_events(response.content)
    assert events[0].data.startswith("[[[")  # relayed: no PII, nothing to redact
    assert json.loads(events[1].data)["choices"][0]["delta"]["content"] == "ok [REDACTED]"
    assert events[-1].data == "[DONE]"


# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------


async def test_pattern_subset_and_replacement_token(
    gateway_for: Factory, make_settings: Settings, openai_text: Text
) -> None:
    settings = make_settings(pattern_names=("email",), replacement="<pii>")
    async with gateway_for(settings) as client:
        response = await client.post("/v1/chat/completions", json=chat(chunk_size=5))
    text = openai_text(response.content)
    assert "<pii>" in text
    assert "536-22-8041" in text  # SSN detector disabled on purpose
    assert "@" not in text


# ---------------------------------------------------------------------------
# Round two: authentication forms, header hygiene, error relays, routing errors
# ---------------------------------------------------------------------------


async def test_gateway_key_is_accepted_as_x_api_key_and_never_forwarded(
    gateway_for: Factory, make_settings: Settings, provider: MockProvider
) -> None:
    # No upstream key configured: the gateway must still not leak its own key upstream.
    settings = make_settings(api_keys=("gw-key-1",), upstream_api_key=None)
    async with gateway_for(settings) as client:
        ok = await client.post(
            "/v1/messages", json=messages(stream=False), headers={"x-api-key": "gw-key-1"}
        )
        bad = await client.post(
            "/v1/messages", json=messages(stream=False), headers={"x-api-key": "nope"}
        )
    assert ok.status_code == 200
    assert bad.status_code == 401
    forwarded = provider.state.last_headers
    assert "gw-key-1" not in json.dumps(forwarded)
    assert "authorization" not in forwarded
    assert "x-api-key" not in forwarded


async def test_exactly_one_content_type_header_reaches_the_provider(
    gateway: httpx.AsyncClient, provider: MockProvider
) -> None:
    response = await gateway.post(
        "/v1/chat/completions",
        content=json.dumps(chat(stream=False)),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    names = [name.lower() for name, _ in provider.state.last_raw_headers]
    assert names.count("content-type") == 1
    assert names.count("accept-encoding") == 1


async def test_rate_limit_headers_and_sse_error_bodies_are_relayed(
    gateway_for: Factory, make_settings: Settings
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messages"):
            body = (
                b"event: error\ndata: "
                b'{"type":"error","error":{"type":"overloaded_error","message":"busy"}}\n\n'
            )
            return httpx.Response(529, headers={"content-type": "text/event-stream"}, content=body)
        return httpx.Response(
            429,
            headers={
                "retry-after": "7",
                "x-ratelimit-remaining-requests": "0",
                "x-internal-node": "node-7",
            },
            json={"error": {"message": "slow down", "type": "rate_limit_error"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), client) as gateway:
        limited = await gateway.post("/v1/chat/completions", json=chat(stream=False))
        overloaded = await gateway.post("/v1/messages", json=messages())
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "7"
    assert limited.headers["x-ratelimit-remaining-requests"] == "0"
    assert "x-internal-node" not in limited.headers
    assert limited.json()["error"]["type"] == "rate_limit_error"
    assert overloaded.status_code == 529
    assert overloaded.json()["error"]["type"] == "overloaded_error"


async def test_routing_errors_use_the_standard_error_shape(gateway: httpx.AsyncClient) -> None:
    missing = await gateway.get("/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["type"] == "not_found"
    wrong_method = await gateway.put("/v1/messages")
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["type"] == "method_not_allowed"
    assert "POST" in wrong_method.headers["allow"]
    trailing = await gateway.post("/v1/messages/", json=messages(stream=False))
    assert trailing.status_code == 404  # no redirect built from the Host header


async def test_provider_json_with_nan_fails_closed(
    gateway_for: Factory, make_settings: Settings
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"choices": [{"message": {"content": "x"}}], "usage": {"total": NaN}}',
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat(stream=False))
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_unparseable"


# ---------------------------------------------------------------------------
# Gemini dialect
# ---------------------------------------------------------------------------


def gemini_body(text: str | None = None, **mock: Any) -> dict[str, Any]:
    prompt = f"echo:{text}" if text is not None else "hello"
    body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if mock:
        body["mock"] = mock
    return body


def _gemini_candidate_text(payload: Any) -> str:
    chunks = payload if isinstance(payload, list) else [payload]
    return "".join(
        part.get("text", "")
        for chunk in chunks
        for candidate in chunk.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
    )


@pytest.mark.parametrize("size", [1, 3, 7, 64])
async def test_gemini_sse_streaming_is_redacted_exactly(
    gateway: httpx.AsyncClient, sse_events: Events, provider: MockProvider, size: int
) -> None:
    response = await gateway.post(
        "/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse",
        json=gemini_body(chunk_size=size),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "alt=sse" in provider.state.last_query  # the query reached the upstream
    text = "".join(_gemini_candidate_text(json.loads(e.data)) for e in sse_events(response.content))
    assert text == redact_text(CANNED_TEXT)


async def test_gemini_array_streaming_and_unary_responses(
    gateway: httpx.AsyncClient,
) -> None:
    array = await gateway.post(
        "/v1beta/models/gemini-2.5-pro:streamGenerateContent", json=gemini_body(chunk_size=5)
    )
    assert array.status_code == 200
    assert isinstance(array.json(), list)
    assert _gemini_candidate_text(array.json()) == redact_text(CANNED_TEXT)

    unary = await gateway.post("/v1/models/gemini-2.5-pro:generateContent", json=gemini_body())
    assert unary.status_code == 200
    assert _gemini_candidate_text(unary.json()) == redact_text(CANNED_TEXT)

    tool = await gateway.post(
        "/v1beta/models/gemini-2.5-pro:generateContent", json=gemini_body(mode="tool_call")
    )
    args = tool.json()["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]
    assert args == json.loads(redact_text(TOOL_ARGUMENTS))


async def test_gemini_unknown_actions_and_versions_are_not_proxied(
    gateway: httpx.AsyncClient, provider: MockProvider
) -> None:
    for path in (
        "/v1beta/models/gemini-2.5-pro:countTokens",
        "/v2/models/gemini-2.5-pro:generateContent",
        "/v1beta/models/gemini-2.5-pro",
    ):
        response = await gateway.post(path, json=gemini_body())
        assert response.status_code == 404, path
        assert response.json()["error"]["type"] == "not_found"
    assert provider.state.requests == 0


async def test_gemini_clients_authenticate_with_their_own_header_or_query_key(
    gateway_for: Factory, make_settings: Settings, provider: MockProvider
) -> None:
    async with gateway_for(make_settings(api_keys=("gw-key-1",))) as client:
        by_header = await client.post(
            "/v1beta/models/m:generateContent",
            json=gemini_body(),
            headers={"x-goog-api-key": "gw-key-1"},
        )
        by_query = await client.post(
            "/v1beta/models/m:generateContent?key=gw-key-1&alt=json", json=gemini_body()
        )
        wrong = await client.post("/v1beta/models/m:generateContent?key=nope", json=gemini_body())
    assert by_header.status_code == 200
    assert by_query.status_code == 200
    assert wrong.status_code == 401
    # The gateway key never travels upstream, neither as a header nor as a query parameter.
    assert "gw-key-1" not in provider.state.last_query
    assert "key=" not in provider.state.last_query
    assert "alt=json" in provider.state.last_query
    forwarded_key = provider.state.last_headers.get("x-goog-api-key")
    assert forwarded_key  # the upstream key was injected
    assert forwarded_key != "gw-key-1"


async def test_bring_your_own_key_mode_passes_gemini_credentials_through(
    gateway_for: Factory, make_settings: Settings, provider: MockProvider
) -> None:
    async with gateway_for(make_settings(upstream_api_key=None)) as client:
        response = await client.post(
            "/v1beta/models/m:generateContent?key=client-key",
            json=gemini_body(),
            headers={"x-goog-api-key": "client-header-key"},
        )
    assert response.status_code == 200
    assert provider.state.last_headers["x-goog-api-key"] == "client-header-key"
    assert "key=client-key" in provider.state.last_query


# ---------------------------------------------------------------------------
# Upstream statuses, error bodies, stream budgets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [200, 201, 202, 206, 299])
async def test_every_success_status_is_redacted(
    gateway_for: Factory, make_settings: Settings, status: int
) -> None:
    document = {"choices": [{"message": {"content": "card 4111 1111 1111 1111"}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=document)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat(stream=False))
    assert response.status_code == status
    assert response.json()["choices"][0]["message"]["content"] == "card [REDACTED]"


async def test_success_status_with_an_event_stream_is_redacted_too(
    gateway_for: Factory, make_settings: Settings
) -> None:
    body = (
        b'data: {"choices":[{"index":0,"delta":{"content":"mail a@b.io "},'
        b'"finish_reason":null}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, headers={"content-type": "text/event-stream"}, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat())
    assert response.status_code == 200
    assert "a@b.io" not in response.text
    assert "[REDACTED]" in response.text


async def test_provider_error_documents_are_redacted_before_relay(
    gateway_for: Factory, make_settings: Settings
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": "bad prompt from ops@example.com", "type": "x"}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat(stream=False))
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "bad prompt from [REDACTED]"


async def test_stream_budgets_end_runaway_streams(
    gateway_for: Factory, make_settings: Settings, sse_events: Events
) -> None:
    async def endless() -> Any:
        while True:
            yield b'data: {"choices":[{"index":0,"delta":{"content":"word "}}]}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_AsyncByteStream(endless()),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with gateway_for(make_settings(max_stream_bytes=20_000), client) as gateway:
        response = await gateway.post("/v1/chat/completions", json=chat())
    events = sse_events(response.content)
    assert events[-1].event == "error"
    assert json.loads(events[-1].data)["error"]["type"] == "upstream_stream_error"
    assert len(response.content) < 60_000
