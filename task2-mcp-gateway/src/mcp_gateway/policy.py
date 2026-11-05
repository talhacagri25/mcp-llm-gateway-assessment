"""Method level authorisation policy for MCP tool calls.

The policy is deliberately small and data driven: a list of rules mapping a
tool name prefix to the roles allowed to call it. Anything that is not a
``tools/call`` is outside the policy's scope and forwarded; a ``tools/call``
whose tool name cannot be determined is treated as invalid rather than
allowed (fail closed).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from mcp_gateway.auth import Principal
from mcp_gateway.jsonrpc import MAX_NAME_LENGTH, JsonRpcMessage, clean_identifier

TOOLS_CALL = "tools/call"


def canonical(text: str) -> str:
    """Normalise a method or tool name for comparison.

    NFKC folds full-width and other compatibility look-alikes, format
    characters (zero-width spaces, joiners, bidi marks) are dropped, and the
    result is case folded and trimmed. A downstream that normalises names must
    never see a request the gateway considered unrelated to ``admin_``.
    """
    folded = unicodedata.normalize("NFKC", text)
    stripped = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    return stripped.casefold().strip()


def is_tools_call(method: str | None) -> bool:
    return method is not None and canonical(method) == TOOLS_CALL


class Verdict(Enum):
    ALLOW = "allow"
    DENY = "deny"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class PrefixRule:
    prefix: str
    allowed_roles: frozenset[str]

    def matches(self, tool_name: str) -> bool:
        # Compared in canonical form on purpose: a downstream that normalises
        # tool names must not become a bypass. Exact-match downstreams simply
        # return "unknown tool" for odd spellings, so nothing legitimate is lost.
        return canonical(tool_name).startswith(canonical(self.prefix))


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    reason: str
    tool: str | None = None
    required_roles: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""


class ToolAccessPolicy:
    def __init__(self, rules: Sequence[PrefixRule]) -> None:
        self._rules = tuple(rules)
        self._canonical_prefixes = tuple((rule, canonical(rule.prefix)) for rule in rules)

    @classmethod
    def default(
        cls,
        admin_tool_prefixes: Iterable[str] = ("admin_",),
        admin_roles: Iterable[str] = ("admin",),
    ) -> ToolAccessPolicy:
        roles = frozenset(admin_roles)
        return cls([PrefixRule(prefix=p, allowed_roles=roles) for p in admin_tool_prefixes])

    @property
    def rules(self) -> tuple[PrefixRule, ...]:
        return self._rules

    def evaluate(self, principal: Principal, message: JsonRpcMessage) -> Decision:
        if not is_tools_call(message.method):
            return Decision(Verdict.ALLOW, "method_not_restricted")

        params = message.params
        if not isinstance(params, dict):
            return Decision(
                Verdict.INVALID,
                "params_not_object",
                message="Invalid params: tools/call requires a params object",
            )
        name = params.get("name")
        if not isinstance(name, str) or not name.strip():
            return Decision(
                Verdict.INVALID,
                "tool_name_missing",
                message="Invalid params: tools/call requires params.name as a non-empty string",
            )
        # Bounded and checked before any normalisation: the raw value (padding
        # included) is what gets echoed, logged and compared.
        if len(name) > MAX_NAME_LENGTH:
            return Decision(
                Verdict.INVALID,
                "tool_name_too_long",
                message=f"Invalid params: params.name exceeds {MAX_NAME_LENGTH} characters",
            )
        if not clean_identifier(name):
            return Decision(
                Verdict.INVALID,
                "tool_name_invalid",
                message="Invalid params: params.name contains control characters",
            )

        folded = canonical(name)
        for rule, prefix in self._canonical_prefixes:
            if folded.startswith(prefix):
                required = tuple(sorted(rule.allowed_roles))
                if principal.role in rule.allowed_roles:
                    return Decision(
                        Verdict.ALLOW, "role_permitted", tool=name, required_roles=required
                    )
                return Decision(
                    Verdict.DENY,
                    "role_not_permitted",
                    tool=name,
                    required_roles=required,
                    message=(
                        f"Unauthorized Tool Call: '{name}' requires role "
                        f"{' or '.join(required)}, caller has role '{principal.role}'"
                    ),
                )
        return Decision(Verdict.ALLOW, "tool_not_restricted", tool=name)
