"""Provider dialects: how the gateway's request shape maps onto each provider API.

Clients speak the OpenAI Chat Completions request shape to the router. A
provider either speaks the same dialect (OpenAI, Azure OpenAI, vLLM, TGI,
Ollama, Groq, Mistral, OpenRouter and most self-hosted fine-tunes) or needs
translation; the Anthropic Messages API is the major one that does. Responses
are normalised back to the Chat Completions shape, so clients see one contract
whichever provider served them.

Translation is strict rather than lossy: a request that uses a feature the
target dialect cannot express (``n > 1``, log probabilities, logit bias, a
structured ``response_format``) raises ``UnsupportedRequestError`` instead of
being sent with silently changed semantics. The router then tries the next
provider and, when nobody can serve the shape, answers ``400``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

ANTHROPIC_VERSION: Final = "2023-06-01"
DIALECT_NAMES: Final = ("openai", "anthropic")


class UnsupportedRequestError(ValueError):
    """The request uses a feature this provider dialect cannot express."""


DEFAULT_FORWARDED_CLIENT_HEADERS: Final = ("traceparent", "tracestate")
"""Client headers relayed to providers by default: distributed tracing context only."""


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


class ProviderDialect(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def extra_failover_statuses(self) -> frozenset[int]: ...

    def prepare(
        self,
        payload: dict[str, Any],
        *,
        model: str | None,
        api_key: str | None,
        request_id: str,
        default_max_tokens: int,
    ) -> PreparedRequest: ...

    def normalize(self, body: Any, *, requested_model: str | None) -> Any: ...


# --- OpenAI compatible -------------------------------------------------------


MAX_TOKENS_FIELDS: Final = ("max_tokens", "max_completion_tokens")


@dataclass(frozen=True, slots=True)
class OpenAICompatibleDialect:
    name: str = "openai"
    extra_failover_statuses: frozenset[int] = field(default_factory=frozenset)
    max_tokens_field: str = "max_tokens"
    """Field that carries the completion cap when the client sent none (reasoning models
    want ``max_completion_tokens``)."""

    def prepare(
        self,
        payload: dict[str, Any],
        *,
        model: str | None,
        api_key: str | None,
        request_id: str,
        default_max_tokens: int,
    ) -> PreparedRequest:
        body = dict(payload)
        if model:
            body["model"] = model
        if all(body.get(key) is None for key in MAX_TOKENS_FIELDS):
            # The reservation assumed this budget; the provider must be held to it,
            # otherwise its own default output cap decides what the tenant spends.
            body[self.max_tokens_field] = default_max_tokens
        headers = {"content-type": "application/json", "x-request-id": request_id}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        return PreparedRequest("/v1/chat/completions", headers, body)

    def normalize(self, body: Any, *, requested_model: str | None) -> Any:
        return body


# --- Anthropic Messages --------------------------------------------------------

_STOP_REASONS: Final = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _text_of(content: Any, *, what: str) -> str:
    """Plain text of a string or a list of text parts; anything else is unsupported."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                pieces.append(part["text"])
            else:
                raise UnsupportedRequestError(f"{what} may only contain text parts")
        return "".join(pieces)
    raise UnsupportedRequestError(f"{what} must be a string or a list of parts")


def _image_block(part: dict[str, Any]) -> dict[str, Any]:
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    if not isinstance(url, str) or not url:
        raise UnsupportedRequestError("image_url parts need a url")
    if url.startswith("data:"):
        header, separator, data = url.partition(",")
        media_type = header[5:].split(";")[0]
        if not separator or ";base64" not in header or not media_type:
            raise UnsupportedRequestError("data URLs must be base64 encoded with a media type")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        raise UnsupportedRequestError("user content must be a string or a list of parts")
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise UnsupportedRequestError("content parts must be objects")
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            if part["text"]:
                blocks.append({"type": "text", "text": part["text"]})
        elif kind == "image_url":
            blocks.append(_image_block(part))
        else:
            raise UnsupportedRequestError(f"content part type {kind!r} is not supported")
    return blocks


def _assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    text = _text_of(message.get("content"), what="assistant content")
    if text:
        blocks.append({"type": "text", "text": text})
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal:
        blocks.append({"type": "text", "text": refusal})
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise UnsupportedRequestError("tool_calls must be a list")
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise UnsupportedRequestError("tool_calls entries need a function name")
            arguments = function.get("arguments", "{}")
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except ValueError as exc:
                raise UnsupportedRequestError("tool call arguments must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise UnsupportedRequestError("tool call arguments must be a JSON object")
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(call.get("id") or f"call_{len(blocks)}"),
                    "name": function["name"],
                    "input": parsed,
                }
            )
    if message.get("function_call") is not None:
        raise UnsupportedRequestError("legacy function_call messages are not supported")
    return blocks


def _tool_result_block(message: dict[str, Any]) -> dict[str, Any]:
    call_id = message.get("tool_call_id")
    if not isinstance(call_id, str) or not call_id:
        raise UnsupportedRequestError("tool messages need a tool_call_id")
    content = _text_of(message.get("content"), what="tool result content")
    return {"type": "tool_result", "tool_use_id": call_id, "content": content}


def _tools(payload: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    tools = payload.get("tools")
    choice = payload.get("tool_choice")
    if tools is None:
        if choice not in (None, "none", "auto"):
            raise UnsupportedRequestError("tool_choice without tools")
        return None, None
    if not isinstance(tools, list):
        raise UnsupportedRequestError("tools must be a list")
    if choice == "none":
        return None, None  # the model must not call tools: do not offer any
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if tool.get("type", "function") != "function" or not isinstance(function, dict):
            raise UnsupportedRequestError("only function tools are supported")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise UnsupportedRequestError("tools need a function name")
        entry: dict[str, Any] = {
            "name": name,
            "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
        }
        if isinstance(function.get("description"), str):
            entry["description"] = function["description"]
        converted.append(entry)
    tool_choice: dict[str, Any]
    if choice in (None, "auto"):
        tool_choice = {"type": "auto"}
    elif choice == "required":
        tool_choice = {"type": "any"}
    elif isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        tool_choice = {"type": "tool", "name": str(choice["function"].get("name", ""))}
    else:
        raise UnsupportedRequestError(f"tool_choice {str(choice)[:64]!r} is not supported")
    if payload.get("parallel_tool_calls") is False:
        tool_choice["disable_parallel_tool_use"] = True
    return converted, tool_choice


def _check_unsupported_sampling(payload: dict[str, Any]) -> None:
    if payload.get("n") not in (None, 1):
        raise UnsupportedRequestError("n > 1 is not supported by the Anthropic dialect")
    for key in ("logprobs", "top_logprobs", "seed"):
        if payload.get(key) not in (None, False, 0):
            raise UnsupportedRequestError(f"{key} is not supported by the Anthropic dialect")
    if payload.get("logit_bias"):
        raise UnsupportedRequestError("logit_bias is not supported by the Anthropic dialect")
    for key in ("presence_penalty", "frequency_penalty"):
        if payload.get(key) not in (None, 0, 0.0):
            raise UnsupportedRequestError(f"{key} is not supported by the Anthropic dialect")
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") not in (None, "text"):
        raise UnsupportedRequestError("structured response_format is not supported by this dialect")


def translate_chat_request(
    payload: dict[str, Any], *, model: str | None, default_max_tokens: int
) -> dict[str, Any]:
    """Chat Completions request body to Messages API request body."""
    _check_unsupported_sampling(payload)
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            raise UnsupportedRequestError(f"{role} messages must not be empty")
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)  # the Messages API alternates roles
        else:
            messages.append({"role": role, "content": blocks})

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise UnsupportedRequestError("messages are required")
        raw_messages = [{"role": "user", "content": prompt}]
    for message in raw_messages:
        if not isinstance(message, dict):
            raise UnsupportedRequestError("messages must be objects")
        role = message.get("role")
        if role in ("system", "developer"):
            text = _text_of(message.get("content"), what="system content")
            if text:
                system_parts.append(text)
        elif role == "user":
            append("user", _user_blocks(message.get("content")))
        elif role == "assistant":
            append("assistant", _assistant_blocks(message))
        elif role == "tool":
            append("user", [_tool_result_block(message)])
        else:
            raise UnsupportedRequestError(f"message role {role!r} is not supported")
    if not messages:
        raise UnsupportedRequestError("at least one user or assistant message is required")
    if messages[0]["role"] != "user":
        raise UnsupportedRequestError("the conversation must start with a user message")

    budget = None
    for key in ("max_completion_tokens", "max_tokens"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            budget = value
            break
    body: dict[str, Any] = {
        "model": model or payload.get("model"),
        "max_tokens": budget or default_max_tokens,
        "messages": messages,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    for key in ("temperature", "top_p", "top_k"):
        if payload.get(key) is not None:
            body[key] = payload[key]
    stop = payload.get("stop")
    if isinstance(stop, str) and stop:
        body["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        body["stop_sequences"] = [str(s) for s in stop]
    if isinstance(payload.get("user"), str) and payload["user"]:
        body["metadata"] = {"user_id": payload["user"]}
    tools, tool_choice = _tools(payload)
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    if not body["model"]:
        raise UnsupportedRequestError("a model is required")
    return body


def _count(value: Any) -> int:
    """A usage figure, or 0 when the provider sent something that is not a count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def normalize_messages_response(body: Any, *, requested_model: str | None) -> Any:
    """Messages API response to a Chat Completions response; ``None`` when not a message."""
    if not isinstance(body, dict) or not isinstance(body.get("content"), list):
        return None
    text_parts: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in body["content"]:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif kind == "thinking" and isinstance(block.get("thinking"), str):
            reasoning.append(block["thinking"])
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    raw_usage = body.get("usage")
    usage_in: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = sum(
        _count(usage_in.get(key))
        for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    completion_tokens = _count(usage_in.get("output_tokens"))
    return {
        "id": str(body.get("id", "")),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or requested_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _STOP_REASONS.get(str(body.get("stop_reason")), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@dataclass(frozen=True, slots=True)
class AnthropicMessagesDialect:
    name: str = "anthropic"
    extra_failover_statuses: frozenset[int] = frozenset({529})

    def prepare(
        self,
        payload: dict[str, Any],
        *,
        model: str | None,
        api_key: str | None,
        request_id: str,
        default_max_tokens: int,
    ) -> PreparedRequest:
        body = translate_chat_request(payload, model=model, default_max_tokens=default_max_tokens)
        headers = {
            "content-type": "application/json",
            "x-request-id": request_id,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if api_key:
            headers["x-api-key"] = api_key
        return PreparedRequest("/v1/messages", headers, body)

    def normalize(self, body: Any, *, requested_model: str | None) -> Any:
        return normalize_messages_response(body, requested_model=requested_model)


def dialect_for(name: str, *, max_tokens_field: str = "max_tokens") -> ProviderDialect:
    if name == "openai":
        return OpenAICompatibleDialect(max_tokens_field=max_tokens_field)
    if name == "anthropic":
        return AnthropicMessagesDialect()
    raise ValueError(f"unknown provider dialect {name!r}")
