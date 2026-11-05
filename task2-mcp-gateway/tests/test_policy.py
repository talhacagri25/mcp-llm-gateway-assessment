"""Tool access policy decisions."""

from __future__ import annotations

import json
from typing import Any

import pytest

from mcp_gateway.auth import Principal
from mcp_gateway.jsonrpc import parse_payload
from mcp_gateway.policy import PrefixRule, ToolAccessPolicy, Verdict

ADMIN = Principal("alice", "admin", "fp")
VIEWER = Principal("bob", "viewer", "fp")


def message(method: str, params: Any = None) -> Any:
    frame: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        frame["params"] = params
    return parse_payload(json.dumps(frame).encode()).messages[0]


@pytest.fixture
def policy() -> ToolAccessPolicy:
    return ToolAccessPolicy.default()


@pytest.mark.parametrize("method", ["tools/list", "initialize", "ping", "resources/read"])
def test_non_tool_call_methods_are_not_restricted(policy: ToolAccessPolicy, method: str) -> None:
    decision = policy.evaluate(VIEWER, message(method, {"name": "admin_reset_key"}))
    assert decision.verdict is Verdict.ALLOW
    assert decision.reason == "method_not_restricted"


def test_viewer_may_call_unrestricted_tool(policy: ToolAccessPolicy) -> None:
    decision = policy.evaluate(VIEWER, message("tools/call", {"name": "get_weather"}))
    assert decision.verdict is Verdict.ALLOW
    assert decision.tool == "get_weather"


@pytest.mark.parametrize(
    "name", ["admin_reset_key", "admin_", "ADMIN_RESET_KEY", "Admin_Delete", " admin_reset_key"]
)
def test_viewer_is_denied_admin_tools_case_insensitively(
    policy: ToolAccessPolicy, name: str
) -> None:
    decision = policy.evaluate(VIEWER, message("tools/call", {"name": name}))
    assert decision.verdict is Verdict.DENY
    assert decision.required_roles == ("admin",)
    assert "Unauthorized Tool Call" in decision.message


@pytest.mark.parametrize("name", ["administrator", "my_admin_tool", "adminreset", "ad_min_x"])
def test_prefix_is_matched_at_the_start_only(policy: ToolAccessPolicy, name: str) -> None:
    assert policy.evaluate(VIEWER, message("tools/call", {"name": name})).verdict is Verdict.ALLOW


def test_admin_may_call_admin_tools(policy: ToolAccessPolicy) -> None:
    decision = policy.evaluate(ADMIN, message("tools/call", {"name": "admin_reset_key"}))
    assert decision.verdict is Verdict.ALLOW
    assert decision.reason == "role_permitted"


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        (None, "params_not_object"),
        ([], "params_not_object"),
        ({}, "tool_name_missing"),
        ({"name": 5}, "tool_name_missing"),
        ({"name": None}, "tool_name_missing"),
        ({"name": ""}, "tool_name_missing"),
        ({"name": "   "}, "tool_name_missing"),
        ({"name": ["admin_reset_key"]}, "tool_name_missing"),
    ],
)
def test_undeterminable_tool_name_fails_closed(
    policy: ToolAccessPolicy, params: Any, reason: str
) -> None:
    decision = policy.evaluate(ADMIN, message("tools/call", params))
    assert decision.verdict is Verdict.INVALID
    assert decision.reason == reason


def test_custom_rules() -> None:
    policy = ToolAccessPolicy(
        [
            PrefixRule("admin_", frozenset({"admin"})),
            PrefixRule("ops_", frozenset({"admin", "operator"})),
        ]
    )
    operator = Principal("carol", "operator", "fp")
    assert (
        policy.evaluate(operator, message("tools/call", {"name": "ops_restart"})).verdict
        is Verdict.ALLOW
    )
    assert (
        policy.evaluate(operator, message("tools/call", {"name": "admin_x"})).verdict
        is Verdict.DENY
    )
    assert (
        policy.evaluate(VIEWER, message("tools/call", {"name": "ops_restart"})).verdict
        is Verdict.DENY
    )


def test_default_factory_accepts_multiple_prefixes_and_roles() -> None:
    policy = ToolAccessPolicy.default(("admin_", "danger_"), ("admin", "sre"))
    sre = Principal("dan", "sre", "fp")
    assert (
        policy.evaluate(sre, message("tools/call", {"name": "danger_wipe"})).verdict
        is Verdict.ALLOW
    )
    assert (
        policy.evaluate(VIEWER, message("tools/call", {"name": "danger_wipe"})).verdict
        is Verdict.DENY
    )


@pytest.mark.parametrize("method", ["Tools/Call", " tools/call ", "tools/call\u200b", "TOOLS/CALL"])
def test_method_look_alikes_are_still_policed(policy: ToolAccessPolicy, method: str) -> None:
    decision = policy.evaluate(VIEWER, message(method, {"name": "admin_reset_key"}))
    assert decision.verdict is Verdict.DENY


@pytest.mark.parametrize(
    "name",
    [
        "\uff21dmin_reset_key",  # full-width A
        "\u200badmin_reset_key",  # zero-width space in front
        "ad\u200bmin_reset_key",  # zero-width space inside
        "\u202eadmin_reset_key",  # bidi override
        "ADMIN_RESET_KEY",
    ],
)
def test_tool_name_look_alikes_are_denied(policy: ToolAccessPolicy, name: str) -> None:
    decision = policy.evaluate(VIEWER, message("tools/call", {"name": name}))
    assert decision.verdict is Verdict.DENY


def test_oversized_and_control_laden_tool_names_are_invalid(policy: ToolAccessPolicy) -> None:
    padded = " " * 300_000 + "admin_reset_key"
    decision = policy.evaluate(VIEWER, message("tools/call", {"name": padded}))
    assert decision.verdict is Verdict.INVALID
    assert decision.reason == "tool_name_too_long"
    assert decision.tool is None  # nothing of the oversized value is echoed back
    control = policy.evaluate(VIEWER, message("tools/call", {"name": "admin\x00_reset"}))
    assert control.verdict is Verdict.INVALID
    assert control.reason == "tool_name_invalid"
    exact = policy.evaluate(VIEWER, message("tools/call", {"name": "x" * 256}))
    assert exact.verdict is Verdict.ALLOW
