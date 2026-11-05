"""JSON-RPC 2.0 wire format parsing."""

from __future__ import annotations

import json
from typing import Any

import pytest

from mcp_gateway.jsonrpc import (
    INVALID_REQUEST,
    PARSE_ERROR,
    JsonRpcProtocolError,
    error_response,
    parse_payload,
)


def _parse(obj: Any) -> Any:
    return parse_payload(json.dumps(obj).encode())


def test_single_request() -> None:
    payload = _parse({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "x"}})
    assert payload.is_batch is False
    (message,) = payload.messages
    assert message.is_request
    assert not message.is_notification
    assert message.request_id == 7
    assert message.method == "tools/call"
    assert message.params == {"name": "x"}


def test_notification_has_no_id() -> None:
    (message,) = _parse({"jsonrpc": "2.0", "method": "notifications/initialized"}).messages
    assert message.is_notification
    assert message.has_id is False
    assert message.request_id is None


def test_null_id_is_a_request_not_a_notification() -> None:
    (message,) = _parse({"jsonrpc": "2.0", "id": None, "method": "ping"}).messages
    assert message.has_id is True
    assert message.is_request


@pytest.mark.parametrize("request_id", ["abc", 0, 1.5, ""])
def test_id_types_are_preserved(request_id: Any) -> None:
    (message,) = _parse({"jsonrpc": "2.0", "id": request_id, "method": "ping"}).messages
    assert message.request_id == request_id


def test_client_to_server_response_is_accepted() -> None:
    (message,) = _parse({"jsonrpc": "2.0", "id": 3, "result": {}}).messages
    assert message.is_response
    assert message.method is None


def test_batch() -> None:
    payload = _parse(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/x"},
        ]
    )
    assert payload.is_batch
    assert len(payload.messages) == 2
    assert [m.request_id for m in payload.requests] == [1]


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"{",
        b"\xff\xfe",
        b"nope",
        b"[1,",
        b"[" * 200_000,
        b'{"a":' * 200_000,
        b'{"jsonrpc":"2.0","id":NaN,"method":"x"}',
        b'{"jsonrpc":"2.0","id":1,"method":"x","params":{"v":Infinity}}',
        b'{"jsonrpc":"2.0","id":' + b"1" * 5000 + b',"method":"x"}',
    ],
)
def test_parse_errors(body: bytes) -> None:
    with pytest.raises(JsonRpcProtocolError) as exc_info:
        parse_payload(body)
    assert exc_info.value.code == PARSE_ERROR
    assert exc_info.value.request_id is None


@pytest.mark.parametrize(
    ("obj", "expected_id"),
    [
        (42, None),
        ("string", None),
        ([], None),
        ([1], None),
        ({"id": 1, "method": "x"}, 1),  # missing jsonrpc
        ({"jsonrpc": "1.0", "id": 1, "method": "x"}, 1),
        ({"jsonrpc": "2.0", "id": 1}, 1),  # no method, result or error
        ({"jsonrpc": "2.0", "id": 1, "method": ""}, 1),
        ({"jsonrpc": "2.0", "id": 1, "method": 5}, 1),
        ({"jsonrpc": "2.0", "id": 1, "method": "x", "params": "str"}, 1),
        ({"jsonrpc": "2.0", "id": 1, "method": "x", "params": 3}, 1),
        ({"jsonrpc": "2.0", "id": True, "method": "x"}, None),
        ({"jsonrpc": "2.0", "id": {"a": 1}, "method": "x"}, None),
        ({"jsonrpc": "2.0", "id": [1], "method": "x"}, None),
    ],
)
def test_invalid_requests(obj: Any, expected_id: Any) -> None:
    with pytest.raises(JsonRpcProtocolError) as exc_info:
        _parse(obj)
    assert exc_info.value.code == INVALID_REQUEST
    assert exc_info.value.request_id == expected_id


def test_batch_with_one_malformed_element_is_rejected() -> None:
    with pytest.raises(JsonRpcProtocolError) as exc_info:
        _parse([{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"bogus": True}])
    assert exc_info.value.code == INVALID_REQUEST


def test_error_response_shape() -> None:
    assert error_response("abc", -32001, "Unauthorized Tool Call") == {
        "jsonrpc": "2.0",
        "id": "abc",
        "error": {"code": -32001, "message": "Unauthorized Tool Call"},
    }
    with_data = error_response(None, -32602, "bad", {"reason": "x"})
    assert with_data["error"]["data"] == {"reason": "x"}


@pytest.mark.parametrize(
    "body",
    [
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"a","name":"b"}}',
        b'{"jsonrpc":"2.0","id":1,"method":"ping","method":"tools/call"}',
        b'[{"jsonrpc":"2.0","id":1,"method":"ping","id":2}]',
    ],
)
def test_duplicate_members_are_invalid_requests(body: bytes) -> None:
    with pytest.raises(JsonRpcProtocolError) as exc_info:
        parse_payload(body)
    assert exc_info.value.code == INVALID_REQUEST
    assert "duplicate" in exc_info.value.message


def test_non_finite_numeric_ids_are_invalid() -> None:
    with pytest.raises(JsonRpcProtocolError) as exc_info:
        parse_payload(b'{"jsonrpc":"2.0","id":1e400,"method":"x"}')
    assert exc_info.value.code == INVALID_REQUEST
    assert exc_info.value.request_id is None


def test_batch_size_cap_is_enforced_by_the_parser() -> None:
    three = json.dumps([{"jsonrpc": "2.0", "id": i, "method": "x"} for i in (1, 2, 3)]).encode()
    assert len(parse_payload(three, max_batch_size=3).messages) == 3
    with pytest.raises(JsonRpcProtocolError) as exc_info:
        parse_payload(three, max_batch_size=2)
    assert exc_info.value.code == INVALID_REQUEST
    assert "exceeds 2" in exc_info.value.message


@pytest.mark.parametrize(
    ("body", "code", "fragment"),
    [
        ('{"jsonrpc":"2.0","id":1,"method":"ping"}'.encode("utf-16"), PARSE_ERROR, "UTF-8"),
        (b"\xff\xfe{}", PARSE_ERROR, "UTF-8"),
        ("\ufeff{}".encode(), PARSE_ERROR, "Parse error"),
        (
            b'{"jsonrpc":"2.0","id":1,"method":"' + b"m" * 257 + b'"}',
            INVALID_REQUEST,
            "exceeds 256",
        ),
        (
            b'{"jsonrpc":"2.0","id":1,"method":"tools\\u0000/call"}',
            INVALID_REQUEST,
            "control characters",
        ),
        (
            b'{"jsonrpc":"2.0","id":1,"method":"tools\\ud800/call"}',
            INVALID_REQUEST,
            "control characters",
        ),
    ],
)
def test_encodings_and_hostile_method_names_are_refused(
    body: bytes, code: int, fragment: str
) -> None:
    with pytest.raises(JsonRpcProtocolError) as exc_info:
        parse_payload(body)
    assert exc_info.value.code == code
    assert fragment in exc_info.value.message


def test_method_names_at_the_bound_are_accepted() -> None:
    (message,) = _parse({"jsonrpc": "2.0", "id": 1, "method": "m" * 256}).messages
    assert message.method == "m" * 256
