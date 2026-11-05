"""Provider dialect translation between the Chat Completions and Messages shapes."""

from __future__ import annotations

import json
from typing import Any

import pytest

from model_router.dialects import (
    AnthropicMessagesDialect,
    OpenAICompatibleDialect,
    UnsupportedRequestError,
    dialect_for,
    normalize_messages_response,
    translate_chat_request,
)


def test_openai_dialect_pins_the_model_and_sets_the_bearer_header() -> None:
    prepared = OpenAICompatibleDialect().prepare(
        {"model": "client", "messages": []},
        model="pinned",
        api_key="k",
        request_id="r",
        default_max_tokens=10,
    )
    assert prepared.path == "/v1/chat/completions"
    assert prepared.body["model"] == "pinned"
    assert prepared.body["max_tokens"] == 10  # the reserved budget is enforced upstream
    assert prepared.headers["authorization"] == "Bearer k"
    assert OpenAICompatibleDialect().normalize({"x": 1}, requested_model=None) == {"x": 1}
    with pytest.raises(ValueError, match="unknown provider dialect"):
        dialect_for("cohere")


def test_translation_of_a_full_conversation() -> None:
    payload = {
        "model": "gpt-x",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "developer", "content": [{"type": "text", "text": "Use metric units."}]},
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "user", "content": [{"type": "text", "text": "Celsius please."}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "18C, cloudy"},
            {"role": "assistant", "content": "It is 18C."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "image_url", "image_url": {"url": "https://x.example/a.png"}},
                ],
            },
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["END", "STOP"],
        "user": "u-42",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Look up weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    body = translate_chat_request(payload, model="claude-x", default_max_tokens=1024)
    assert body["model"] == "claude-x"
    assert body["max_tokens"] == 300
    assert body["system"] == "Be brief.\n\nUse metric units."
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]  # alternating, merged
    first_user = body["messages"][0]["content"]
    assert [b["text"] for b in first_user] == ["Weather in Paris?", "Celsius please."]
    tool_use = body["messages"][1]["content"][0]
    assert tool_use == {
        "type": "tool_use",
        "id": "call_1",
        "name": "weather",
        "input": {"city": "Paris"},
    }
    assert body["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "18C, cloudy",
    }
    images = body["messages"][4]["content"]
    assert images[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    assert images[2]["source"] == {"type": "url", "url": "https://x.example/a.png"}
    assert body["stop_sequences"] == ["END", "STOP"]
    assert body["metadata"] == {"user_id": "u-42"}
    assert body["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"
    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9


def test_translation_defaults_and_tool_choice_forms() -> None:
    minimal = translate_chat_request(
        {"prompt": "hi", "model": "m"}, model=None, default_max_tokens=77
    )  # prompt-style request
    assert minimal["max_tokens"] == 77
    assert minimal["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    tools = [{"type": "function", "function": {"name": "f"}}]
    base = {"messages": [{"role": "user", "content": "x"}], "tools": tools, "stop": "END"}
    assert translate_chat_request(
        {**base, "tool_choice": "required"}, model="m", default_max_tokens=1
    )["tool_choice"] == {"type": "any"}
    forced = translate_chat_request(
        {**base, "tool_choice": {"type": "function", "function": {"name": "f"}}},
        model="m",
        default_max_tokens=1,
    )
    assert forced["tool_choice"] == {"type": "tool", "name": "f"}
    assert forced["stop_sequences"] == ["END"]
    assert forced["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    none = translate_chat_request({**base, "tool_choice": "none"}, model="m", default_max_tokens=1)
    assert "tools" not in none


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"messages": [{"role": "user", "content": "x"}], "n": 2}, "n > 1"),
        ({"messages": [{"role": "user", "content": "x"}], "logprobs": True}, "logprobs"),
        ({"messages": [{"role": "user", "content": "x"}], "logit_bias": {"1": 5}}, "logit_bias"),
        ({"messages": [{"role": "user", "content": "x"}], "seed": 7}, "seed"),
        (
            {"messages": [{"role": "user", "content": "x"}], "frequency_penalty": 0.5},
            "frequency_penalty",
        ),
        (
            {
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_object"},
            },
            "response_format",
        ),
        ({"messages": [{"role": "assistant", "content": "x"}]}, "start with a user"),
        ({"messages": [{"role": "user", "content": ""}]}, "must not be empty"),
        ({"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]}, "not supported"),
        (
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "f", "arguments": "{"}}],
                    },
                ]
            },
            "valid JSON",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "function_call": {"name": "f", "arguments": "{}"}},
                ]
            },
            "legacy function_call",
        ),
        (
            {"messages": [{"role": "user", "content": "x"}, {"role": "tool", "content": "r"}]},
            "tool_call_id",
        ),
        ({"messages": [{"role": "function", "content": "x"}]}, "role 'function'"),
        ({"messages": [{"role": "user", "content": "x"}], "tool_choice": "weird"}, "tool_choice"),
        ({"messages": []}, "messages are required"),
        ({"messages": [{"role": "user", "content": "x"}]}, "model is required"),
    ],
)
def test_unsupported_shapes_are_refused_not_degraded(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(UnsupportedRequestError, match=match):
        translate_chat_request(payload, model=payload.get("model"), default_max_tokens=10)


def test_messages_response_is_normalised_to_the_chat_completions_shape() -> None:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [
            {"type": "thinking", "thinking": "Let me think.", "signature": "s"},
            {"type": "text", "text": "Checking "},
            {"type": "text", "text": "now."},
            {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Paris"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 7},
    }
    out = normalize_messages_response(body, requested_model=None)
    assert out["object"] == "chat.completion"
    assert out["model"] == "claude-x"
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Checking now."
    assert choice["message"]["reasoning_content"] == "Let me think."
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert out["usage"] == {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22}

    plain = normalize_messages_response(
        {"content": [{"type": "text", "text": "hi"}], "stop_reason": "max_tokens"},
        requested_model="fallback-model",
    )
    assert plain["choices"][0]["finish_reason"] == "length"
    assert plain["model"] == "fallback-model"
    assert normalize_messages_response({"error": {}}, requested_model=None) is None
    assert normalize_messages_response("nope", requested_model=None) is None


def test_anthropic_dialect_headers_and_failover_statuses() -> None:
    dialect = AnthropicMessagesDialect()
    prepared = dialect.prepare(
        {"messages": [{"role": "user", "content": "x"}]},
        model="claude-x",
        api_key="k",
        request_id="r",
        default_max_tokens=5,
    )
    assert prepared.path == "/v1/messages"
    assert prepared.headers["x-api-key"] == "k"
    assert prepared.headers["anthropic-version"]
    assert "authorization" not in prepared.headers
    assert 529 in dialect.extra_failover_statuses


def test_openai_dialect_only_fills_a_missing_completion_cap() -> None:
    kwargs: dict[str, Any] = {"model": None, "api_key": None, "request_id": "r"}
    explicit = OpenAICompatibleDialect().prepare(
        {"messages": [], "max_tokens": 3}, default_max_tokens=10, **kwargs
    )
    assert explicit.body["max_tokens"] == 3
    modern = OpenAICompatibleDialect().prepare(
        {"messages": [], "max_completion_tokens": 4}, default_max_tokens=10, **kwargs
    )
    assert modern.body["max_completion_tokens"] == 4
    assert "max_tokens" not in modern.body
    reasoning = dialect_for("openai", max_tokens_field="max_completion_tokens").prepare(
        {"messages": []}, default_max_tokens=10, **kwargs
    )
    assert reasoning.body["max_completion_tokens"] == 10
    assert "max_tokens" not in reasoning.body


def test_odd_usage_figures_from_a_messages_response_count_as_zero() -> None:
    body = {
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": "12", "output_tokens": -4, "cache_read_input_tokens": True},
    }
    normalized = normalize_messages_response(body, requested_model="m")
    assert normalized["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with pytest.raises(UnsupportedRequestError) as exc_info:
        translate_chat_request(
            {"messages": [{"role": "user", "content": "x"}], "tools": [], "tool_choice": "x" * 500},
            model="m",
            default_max_tokens=1,
        )
    assert len(str(exc_info.value)) < 120
