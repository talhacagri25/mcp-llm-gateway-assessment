"""A controllable mock LLM provider.

It speaks ``POST /v1/chat/completions`` (OpenAI), ``POST /v1/messages``
(Anthropic) and ``POST /v1beta/models/{model}:generateContent`` plus
``:streamGenerateContent`` (Gemini; SSE with ``?alt=sse``, otherwise a JSON
array of chunks), streaming or not. The text it "generates" is either the canned
PII-laden paragraph below or, when the last user message starts with
``echo:``, the remainder of that message verbatim. Chunking and pacing are
controlled per request so tests can place PII exactly on chunk boundaries,
either through a top-level ``mock`` object in the JSON body (which survives
any proxy, since bodies are forwarded verbatim) or through headers:

* ``mock.chunk_size`` / ``X-Mock-Chunk-Size``: characters per streamed delta (default 4)
* ``mock.delay_ms`` / ``X-Mock-Delay-Ms``: pause between deltas (default 0)
* ``mock.mode`` / ``X-Mock-Mode``: ``text`` (default), ``tool_call`` (streams a
  tool call whose arguments contain PII), ``error`` (HTTP 500 JSON error),
  ``abort`` (closes the stream mid-way without a finish event)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

CANNED_TEXT = (
    "Sure. You can reach our billing lead at maria.santos@example.com or her backup "
    "ops-team+billing@mail.example.org. The customer's SSN on file is 536-22-8041 and "
    "the card used was 4111 1111 1111 1111 (expires 09/27). Order 1234567890 shipped."
)
TOOL_ARGUMENTS = '{"recipient": "ceo@example.com", "ssn": "536-22-8041", "note": "call 555-0100"}'


@dataclass
class ProviderState:
    last_headers: dict[str, str] = field(default_factory=dict)
    last_raw_headers: list[tuple[str, str]] = field(default_factory=list)
    """Every header as received, duplicates included."""
    last_query: str = ""
    requests: int = 0


def _sse(data: Any, event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    return f"{prefix}data: {payload}\n\n".encode()


def _response_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                if isinstance(content, str) and content.startswith("echo:"):
                    return content[len("echo:") :]
                break
    return CANNED_TEXT


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _gemini_as_messages(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose Gemini ``contents`` as OpenAI style messages so ``echo:`` works for every dialect."""
    messages: list[dict[str, Any]] = []
    for content in payload.get("contents", []) if isinstance(payload.get("contents"), list) else []:
        if not isinstance(content, dict):
            continue
        raw_parts = content.get("parts")
        parts: list[Any] = raw_parts if isinstance(raw_parts, list) else []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        role = "user" if content.get("role", "user") == "user" else "assistant"
        messages.append({"role": role, "content": text})
    return {**payload, "messages": messages}


class MockProvider:
    def __init__(self) -> None:
        self.state = ProviderState()

    @staticmethod
    def _controls(request: Request, payload: dict[str, Any]) -> tuple[int, float, str]:
        """Chunking controls from a ``mock`` object in the body, else from ``X-Mock-*`` headers."""
        raw_options = payload.get("mock")
        options: dict[str, Any] = raw_options if isinstance(raw_options, dict) else {}
        size = options.get("chunk_size", request.headers.get("x-mock-chunk-size", 4))
        delay = options.get("delay_ms", request.headers.get("x-mock-delay-ms", 0))
        mode = options.get("mode", request.headers.get("x-mock-mode", "text"))
        return max(1, int(size)), max(0.0, float(delay)) / 1000.0, str(mode)

    async def _record(self, request: Request) -> dict[str, Any]:
        self.state.requests += 1
        self.state.last_headers = dict(request.headers)
        self.state.last_query = request.url.query
        self.state.last_raw_headers = [
            (name.decode("latin-1"), value.decode("latin-1")) for name, value in request.headers.raw
        ]
        body = await request.body()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    # -- OpenAI --------------------------------------------------------------

    async def chat_completions(self, request: Request) -> Response:
        payload = await self._record(request)
        size, delay, mode = self._controls(request, payload)
        if mode == "error":
            return JSONResponse(
                {"error": {"message": "mock failure", "type": "server_error", "code": None}},
                status_code=500,
            )
        text = _response_text(payload)
        model = str(payload.get("model", "mock-gpt"))
        created = int(time.time())
        completion_id = f"chatcmpl-mock-{self.state.requests}"
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": max(1, len(text) // 4),
            "total_tokens": 12 + max(1, len(text) // 4),
        }

        if payload.get("stream") is not True:
            message: dict[str, Any] = {"role": "assistant", "content": text}
            if mode == "tool_call":
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "send_email", "arguments": TOOL_ARGUMENTS},
                        }
                    ],
                }
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if mode == "tool_call" else "stop",
                        }
                    ],
                    "usage": usage,
                }
            )

        def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> bytes:
            body: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            body.update(extra)
            return _sse(body)

        async def stream() -> AsyncIterator[bytes]:
            yield chunk({"role": "assistant", "content": ""})
            if mode == "tool_call":
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "send_email", "arguments": ""},
                            }
                        ]
                    }
                )
                for piece in _chunks(TOOL_ARGUMENTS, size):
                    if delay:
                        await asyncio.sleep(delay)
                    yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]})
                yield chunk({}, "tool_calls")
            else:
                pieces = _chunks(text, size)
                for number, piece in enumerate(pieces, start=1):
                    if delay:
                        await asyncio.sleep(delay)
                    if mode == "abort" and number > len(pieces) // 2:
                        return
                    yield chunk({"content": piece})
                yield chunk({}, "stop")
            include_usage = isinstance(payload.get("stream_options"), dict) and payload[
                "stream_options"
            ].get("include_usage")
            if include_usage:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    }
                )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    # -- Anthropic -----------------------------------------------------------

    async def messages(self, request: Request) -> Response:
        payload = await self._record(request)
        size, delay, mode = self._controls(request, payload)
        if mode == "error":
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": "mock failure"}},
                status_code=500,
            )
        text = _response_text(payload)
        model = str(payload.get("model", "mock-claude"))
        message_id = f"msg_mock_{self.state.requests}"
        usage = {"input_tokens": 12, "output_tokens": max(1, len(text) // 4)}

        if payload.get("stream") is not True:
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            if mode == "tool_call":
                content = [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "send_email",
                        "input": json.loads(TOOL_ARGUMENTS),
                    }
                ]
            return JSONResponse(
                {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": content,
                    "stop_reason": "tool_use" if mode == "tool_call" else "end_turn",
                    "stop_sequence": None,
                    "usage": usage,
                }
            )

        async def stream() -> AsyncIterator[bytes]:
            yield _sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 12, "output_tokens": 0},
                    },
                },
                "message_start",
            )
            yield _sse({"type": "ping"}, "ping")
            if mode == "tool_call":
                yield _sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "send_email",
                            "input": {},
                        },
                    },
                    "content_block_start",
                )
                for piece in _chunks(TOOL_ARGUMENTS, size):
                    if delay:
                        await asyncio.sleep(delay)
                    yield _sse(
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "input_json_delta", "partial_json": piece},
                        },
                        "content_block_delta",
                    )
            else:
                yield _sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    "content_block_start",
                )
                pieces = _chunks(text, size)
                for number, piece in enumerate(pieces, start=1):
                    if delay:
                        await asyncio.sleep(delay)
                    if mode == "abort" and number > len(pieces) // 2:
                        return
                    yield _sse(
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": piece},
                        },
                        "content_block_delta",
                    )
            yield _sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
            yield _sse(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "tool_use" if mode == "tool_call" else "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": usage["output_tokens"]},
                },
                "message_delta",
            )
            yield _sse({"type": "message_stop"}, "message_stop")

        return StreamingResponse(stream(), media_type="text/event-stream")

    # -- Gemini --------------------------------------------------------------

    async def generate_content(self, request: Request) -> Response:
        payload = await self._record(request)
        size, delay, mode = self._controls(request, payload)
        model_action = str(request.path_params.get("model_action", ""))
        model, _, action = model_action.rpartition(":")
        if mode == "error":
            return JSONResponse(
                {"error": {"code": 500, "message": "mock failure", "status": "INTERNAL"}},
                status_code=500,
            )
        text = _response_text(_gemini_as_messages(payload))
        usage = {
            "promptTokenCount": 12,
            "candidatesTokenCount": max(1, len(text) // 4),
            "totalTokenCount": 12 + max(1, len(text) // 4),
        }

        def chunk(parts: list[dict[str, Any]], finish: str | None = None) -> dict[str, Any]:
            candidate: dict[str, Any] = {"content": {"parts": parts, "role": "model"}, "index": 0}
            if finish:
                candidate["finishReason"] = finish
            body: dict[str, Any] = {"candidates": [candidate], "modelVersion": model or "mock"}
            if finish:
                body["usageMetadata"] = usage
            return body

        if mode == "tool_call":
            parts: list[dict[str, Any]] = [
                {"functionCall": {"name": "send_email", "args": json.loads(TOOL_ARGUMENTS)}}
            ]
            chunks = [chunk(parts, "STOP")]
        else:
            pieces = _chunks(text, size)
            chunks = [chunk([{"text": piece}]) for piece in pieces]
            if mode == "abort":
                chunks = chunks[: max(1, len(chunks) // 2)]
            else:
                chunks[-1]["candidates"][0]["finishReason"] = "STOP"
                chunks[-1]["usageMetadata"] = usage

        if action == "generateContent":
            merged = chunk([{"text": text}], "STOP")
            if mode == "tool_call":
                merged = chunks[0]
            return JSONResponse(merged)

        if request.query_params.get("alt") == "sse":

            async def stream() -> AsyncIterator[bytes]:
                for body in chunks:
                    if delay:
                        await asyncio.sleep(delay)
                    yield _sse(body)

            return StreamingResponse(stream(), media_type="text/event-stream")

        async def array_stream() -> AsyncIterator[bytes]:
            # The non-SSE streaming shape: a JSON array delivered in pieces.
            yield b"["
            for number, body in enumerate(chunks):
                if delay:
                    await asyncio.sleep(delay)
                prefix = b"," if number else b""
                yield prefix + json.dumps(body, separators=(",", ":")).encode()
            yield b"]"

        return StreamingResponse(array_stream(), media_type="application/json")

    async def last_request(self, request: Request) -> Response:
        return JSONResponse({"requests": self.state.requests, "headers": self.state.last_headers})

    async def health(self, request: Request) -> Response:
        return JSONResponse({"status": "ok", "service": "mock-llm-provider"})


def create_app() -> Starlette:
    provider = MockProvider()
    app = Starlette(
        routes=[
            Route("/v1/chat/completions", provider.chat_completions, methods=["POST"]),
            Route("/v1/messages", provider.messages, methods=["POST"]),
            Route("/{version}/models/{model_action}", provider.generate_content, methods=["POST"]),
            Route("/__mock/last-request", provider.last_request, methods=["GET"]),
            Route("/healthz", provider.health, methods=["GET"]),
        ]
    )
    app.state.provider = provider
    return app
