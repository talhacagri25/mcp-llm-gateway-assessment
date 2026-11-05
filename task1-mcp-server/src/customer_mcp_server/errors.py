"""Domain level errors raised by the customer service.

These represent business rule violations for syntactically valid requests.
Following the MCP specification they are reported *inside* the tool result
with ``isError: true`` so the calling model can read them and self-correct.
They are deliberately distinct from protocol errors (unknown tool, invalid
arguments), which are surfaced as JSON-RPC error responses.
"""

from __future__ import annotations

from typing import ClassVar


class DomainError(Exception):
    """Base class for business rule violations."""

    code: ClassVar[str] = "domain_error"


class CustomerNotFoundError(DomainError):
    code = "customer_not_found"


class CustomerNotEligibleError(DomainError):
    code = "customer_not_eligible"


class RefundExceedsBalanceError(DomainError):
    code = "refund_exceeds_balance"
