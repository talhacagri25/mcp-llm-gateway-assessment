"""Tool input and output schemas.

Input models run in pydantic *strict* mode. The JSON-RPC layer hands the
server already parsed JSON, so there is never a legitimate reason to coerce
types: the string ``"100"`` must not silently become the number ``100`` and
``true`` must not become ``1.0``. Unknown fields are rejected as well, which
stops typos such as ``customerId`` from being ignored and leaving the
required field unset without a clear message.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CUSTOMER_ID_PATTERN: Final[str] = r"^CUST-[0-9]{5}$"
"""Customer identifiers are the literal prefix ``CUST-`` followed by exactly five ASCII digits.

The explicit ``[0-9]`` class matters: ``\\d`` matches every Unicode decimal digit
(Arabic-Indic, fullwidth and so on) in both the Python and the Rust regex engines,
which would let ``CUST-\uff11\uff12\uff13\uff14\uff15`` pass validation.
"""

REASON_MIN_LENGTH: Final[int] = 10
REASON_MAX_LENGTH: Final[int] = 500
REASON_MAX_RAW_LENGTH: Final[int] = 4096
"""Longest raw reason value examined at all, padding included; longer values are refused
before any per-character work, so validation cost is bounded by this constant."""
_CONTROL_CHARACTERS: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
"""Exactly the ``Cc`` category minus tab, newline and carriage return."""
AMOUNT_MAX_DECIMAL_PLACES: Final[int] = 2
MAX_REFUND_AMOUNT: Final[float] = 1_000_000.0
"""Upper bound per refund. A business guard rail, not a protocol constraint."""


class StrictInputModel(BaseModel):
    """Base for every tool input: strict types, no unknown fields, immutable."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class GetCustomerRecordInput(StrictInputModel):
    customer_id: str = Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier in the form CUST-XXXXX where every X is a digit.",
        examples=["CUST-00001"],
    )


class TriggerRefundInput(StrictInputModel):
    customer_id: str = Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier in the form CUST-XXXXX where every X is a digit.",
        examples=["CUST-00001"],
    )
    amount: float = Field(
        gt=0,
        le=MAX_REFUND_AMOUNT,
        allow_inf_nan=False,
        description=(
            "Refund amount in the customer's account currency. Must be a positive, finite "
            f"number with at most {AMOUNT_MAX_DECIMAL_PLACES} decimal places."
        ),
        examples=[49.99],
    )
    reason: str = Field(
        min_length=REASON_MIN_LENGTH,
        max_length=REASON_MAX_LENGTH,
        description=(
            f"Free text justification, {REASON_MIN_LENGTH} to {REASON_MAX_LENGTH} characters "
            "after trimming surrounding whitespace, containing at least one letter or digit."
        ),
        examples=["Customer returned the item within the 30 day window."],
    )

    @field_validator("amount")
    @classmethod
    def _amount_has_at_most_two_decimal_places(cls, value: float) -> float:
        # ``str(float)`` yields the shortest representation that round-trips, so
        # 12.34 becomes "12.34" rather than its binary expansion. This keeps the
        # check meaningful for values a client could have typed.
        exponent = Decimal(str(value)).normalize().as_tuple().exponent
        if isinstance(exponent, int) and exponent < -AMOUNT_MAX_DECIMAL_PLACES:
            raise ValueError(f"amount must have at most {AMOUNT_MAX_DECIMAL_PLACES} decimal places")
        return value

    @field_validator("reason", mode="before")
    @classmethod
    def _reason_trimmed_and_free_of_controls(cls, value: object) -> object:
        # Runs before the length constraints so "10 to 500 characters" means
        # visible text: padding does not count towards either bound. Control
        # characters are checked on the raw value, so none can hide in the
        # whitespace that trimming removes.
        if not isinstance(value, str):
            return value  # the strict type check reports the real problem
        if len(value) > REASON_MAX_RAW_LENGTH:
            raise ValueError(f"reason must not exceed {REASON_MAX_RAW_LENGTH} characters")
        # Only the Cc category is rejected: non-breaking spaces, zero-width joiners
        # inside emoji, bidirectional marks and soft hyphens are legitimate text.
        if _CONTROL_CHARACTERS.search(value):
            raise ValueError("reason must not contain control characters")
        return value.strip()

    @field_validator("reason")
    @classmethod
    def _reason_is_meaningful(cls, value: str) -> str:
        if not any(ch.isalnum() for ch in value):
            raise ValueError("reason must contain at least one letter or digit")
        return value


class CustomerRecord(BaseModel):
    """Public projection of a customer, returned by ``get_customer_record``."""

    model_config = ConfigDict(frozen=True)

    customer_id: str
    full_name: str
    email: str
    tier: Literal["standard", "gold", "platinum"]
    status: Literal["active", "suspended", "closed"]
    refundable_balance: float = Field(description="Amount still eligible for refund.")
    currency: str = Field(description="ISO 4217 currency code of the balance.")


class RefundReceipt(BaseModel):
    """Confirmation returned by ``trigger_refund``."""

    model_config = ConfigDict(frozen=True)

    refund_id: str
    customer_id: str
    amount: float
    currency: str
    reason: str
    status: Literal["accepted"]
    remaining_balance: float
    created_at: datetime
