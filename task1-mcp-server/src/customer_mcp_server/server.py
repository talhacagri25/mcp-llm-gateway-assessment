"""MCP server wiring: tool registry, request handlers and error mapping.

Error mapping follows the MCP specification precisely:

* **Protocol errors** become JSON-RPC error responses. An unknown tool name or
  arguments that fail schema validation are ``-32602 Invalid params``; an
  unexpected exception is ``-32603 Internal error`` with a generic message so
  implementation details never reach the wire.
* **Tool execution errors** (business rule violations such as an unknown
  customer or an insufficient balance) are returned inside a successful
  JSON-RPC response as a ``CallToolResult`` with ``isError: true`` so the
  calling model can read the message and self-correct.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import mcp_types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ValidationError

from customer_mcp_server import __version__
from customer_mcp_server.errors import DomainError
from customer_mcp_server.schemas import (
    CustomerRecord,
    GetCustomerRecordInput,
    RefundReceipt,
    TriggerRefundInput,
)
from customer_mcp_server.service import CustomerService

logger = logging.getLogger(__name__)

SERVER_NAME = "customer-mcp-server"
SERVER_INSTRUCTIONS = (
    "Customer support tools. Look up a customer with get_customer_record before issuing a "
    "refund with trigger_refund. Customer identifiers have the form CUST-XXXXX (five digits)."
)

TOOL_GET_CUSTOMER_RECORD = "get_customer_record"
TOOL_TRIGGER_REFUND = "trigger_refund"

_VALUE_ERROR_PREFIX = "Value error, "
_MAX_ECHOED_CHARS = 80
"""Longest caller supplied fragment echoed back in an error (tool names, field paths)."""
_MAX_REPORTED_ISSUES = 20
"""Validation issues listed per response; the rest are counted, not echoed."""
_MAX_ARGUMENT_KEYS = 32
"""Arguments objects with more members are refused before any per-field validation runs:
a million unknown keys would otherwise cost about sixty times their wire size in memory."""


def _short(text: str, limit: int = _MAX_ECHOED_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Everything the server needs to advertise and execute one tool."""

    name: str
    title: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    annotations: types.ToolAnnotations
    execute: Callable[[Any], BaseModel]

    def definition(self) -> types.Tool:
        return types.Tool(
            name=self.name,
            title=self.title,
            description=self.description,
            input_schema=_json_schema(self.input_model),
            output_schema=_json_schema(self.output_model),
            annotations=self.annotations,
        )


def _json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def _validation_issues(exc: ValidationError) -> list[dict[str, Any]]:
    """Flatten pydantic errors into a compact, input-free list for the error ``data``.

    The submitted values are deliberately not echoed back: they may contain
    personal data and the caller already has them.
    """
    issues: list[dict[str, Any]] = []
    for err in exc.errors(include_url=False, include_input=False, include_context=False):
        message = err["msg"]
        message = message.removeprefix(_VALUE_ERROR_PREFIX)
        path = ".".join(str(part) for part in err["loc"]) or "<root>"
        issues.append({"path": _short(path), "type": err["type"], "message": _short(message)})
    return issues


def _success_result(output: BaseModel) -> types.CallToolResult:
    payload = output.model_dump(mode="json")
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2))],
        structured_content=payload,
    )


def _tool_error_result(exc: DomainError) -> types.CallToolResult:
    message = str(exc)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=f"{exc.code}: {message}")],
        structured_content={"error": {"code": exc.code, "message": message}},
        is_error=True,
    )


class CustomerToolServer:
    """Binds the customer service to an MCP ``Server`` instance."""

    def __init__(self, service: CustomerService) -> None:
        self._service = service
        self._tools: dict[str, ToolSpec] = {spec.name: spec for spec in self._build_specs()}
        self.server: Server[Any] = Server(
            SERVER_NAME,
            version=__version__,
            title="Customer Service MCP Server",
            instructions=SERVER_INSTRUCTIONS,
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
            get_tool_input_schema=self._input_schema_for,
        )

    # -- tool registry -------------------------------------------------------

    def _build_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=TOOL_GET_CUSTOMER_RECORD,
                title="Get customer record",
                description=(
                    "Return the profile and refundable balance of a customer identified by "
                    "a CUST-XXXXX identifier."
                ),
                input_model=GetCustomerRecordInput,
                output_model=CustomerRecord,
                annotations=types.ToolAnnotations(
                    read_only_hint=True,
                    destructive_hint=False,
                    idempotent_hint=True,
                    open_world_hint=False,
                ),
                execute=self._execute_get_customer_record,
            ),
            ToolSpec(
                name=TOOL_TRIGGER_REFUND,
                title="Trigger refund",
                description=(
                    "Issue a refund to an active customer. The amount must be positive with at "
                    "most two decimal places and the reason must be at least ten characters."
                ),
                input_model=TriggerRefundInput,
                output_model=RefundReceipt,
                annotations=types.ToolAnnotations(
                    read_only_hint=False,
                    destructive_hint=True,
                    idempotent_hint=False,
                    open_world_hint=False,
                ),
                execute=self._execute_trigger_refund,
            ),
        ]

    def _execute_get_customer_record(self, args: GetCustomerRecordInput) -> CustomerRecord:
        return self._service.get_customer_record(args.customer_id)

    def _execute_trigger_refund(self, args: TriggerRefundInput) -> RefundReceipt:
        return self._service.trigger_refund(args.customer_id, args.amount, args.reason)

    def _input_schema_for(self, name: str) -> Mapping[str, Any] | None:
        spec = self._tools.get(name)
        return _json_schema(spec.input_model) if spec else None

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    # -- request handlers ----------------------------------------------------

    async def _on_list_tools(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        logger.debug("tools/list", extra={"event": "tools.list", "request_id": ctx.request_id})
        return types.ListToolsResult(tools=[spec.definition() for spec in self._tools.values()])

    async def _on_call_tool(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        spec = self._tools.get(params.name)
        if spec is None:
            logger.warning(
                "unknown tool requested",
                extra={
                    "event": "tools.call.unknown",
                    "tool": params.name,
                    "request_id": ctx.request_id,
                },
            )
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"Unknown tool: {_short(params.name)}",
                data={"tool": _short(params.name), "available_tools": list(self._tools)},
            )

        raw_arguments = params.arguments or {}
        if isinstance(raw_arguments, dict) and len(raw_arguments) > _MAX_ARGUMENT_KEYS:
            logger.warning(
                "tool arguments rejected: too many members",
                extra={
                    "event": "tools.call.invalid_params",
                    "tool": spec.name,
                    "request_id": ctx.request_id,
                    "member_count": len(raw_arguments),
                },
            )
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"Invalid arguments for tool '{spec.name}'",
                data={
                    "tool": spec.name,
                    "issues": [
                        {
                            "path": "<root>",
                            "type": "too_many_fields",
                            "message": f"at most {_MAX_ARGUMENT_KEYS} members are accepted",
                        }
                    ],
                },
            )
        try:
            arguments = spec.input_model.model_validate(raw_arguments)
        except ValidationError as exc:
            issues = _validation_issues(exc)
            reported, omitted = issues[:_MAX_REPORTED_ISSUES], issues[_MAX_REPORTED_ISSUES:]
            logger.warning(
                "tool arguments rejected",
                extra={
                    "event": "tools.call.invalid_params",
                    "tool": spec.name,
                    "request_id": ctx.request_id,
                    "paths": [i["path"] for i in reported],
                    "issue_count": len(issues),
                },
            )
            data: dict[str, Any] = {"tool": spec.name, "issues": reported}
            if omitted:
                # A caller sending thousands of bogus fields gets a bounded answer.
                data["omitted_issues"] = len(omitted)
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"Invalid arguments for tool '{spec.name}'",
                data=data,
            ) from None

        try:
            output = spec.execute(arguments)
        except DomainError as exc:
            logger.info(
                "tool reported a business error",
                extra={
                    "event": "tools.call.tool_error",
                    "tool": spec.name,
                    "error_code": exc.code,
                    "request_id": ctx.request_id,
                },
            )
            return _tool_error_result(exc)
        except Exception:
            # Never let implementation details leak; the full traceback goes to stderr.
            logger.exception(
                "tool raised an unexpected exception",
                extra={
                    "event": "tools.call.failed",
                    "tool": spec.name,
                    "request_id": ctx.request_id,
                },
            )
            raise MCPError(code=types.INTERNAL_ERROR, message="Internal server error") from None

        logger.info(
            "tool call succeeded",
            extra={"event": "tools.call.ok", "tool": spec.name, "request_id": ctx.request_id},
        )
        return _success_result(output)


def build_server(service: CustomerService) -> Server[Any]:
    """Convenience factory returning a ready to run ``Server``."""
    return CustomerToolServer(service).server
