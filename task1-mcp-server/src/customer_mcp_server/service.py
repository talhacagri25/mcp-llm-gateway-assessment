"""Customer domain service with an in-memory repository.

The service is intentionally synchronous: every operation runs without
yielding to the event loop, so concurrent tool calls cannot interleave in
the middle of a balance update. Money is handled as ``Decimal`` internally
and only converted to ``float`` at the schema boundary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Protocol
from uuid import uuid4

from customer_mcp_server.errors import (
    CustomerNotEligibleError,
    CustomerNotFoundError,
    RefundExceedsBalanceError,
)
from customer_mcp_server.schemas import CustomerRecord, RefundReceipt

logger = logging.getLogger(__name__)

_CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class Customer:
    customer_id: str
    full_name: str
    email: str
    tier: Literal["standard", "gold", "platinum"]
    status: Literal["active", "suspended", "closed"]
    refundable_balance: Decimal
    currency: str


class CustomerRepository(Protocol):
    def get(self, customer_id: str) -> Customer | None: ...

    def save(self, customer: Customer) -> None: ...


class InMemoryCustomerRepository:
    """Dictionary backed repository, sufficient for the assessment scope."""

    def __init__(self, customers: Iterable[Customer] = ()) -> None:
        self._customers: dict[str, Customer] = {c.customer_id: c for c in customers}

    @classmethod
    def with_sample_data(cls) -> InMemoryCustomerRepository:
        return cls(
            [
                Customer(
                    "CUST-00001",
                    "Ada Lovelace",
                    "ada@example.com",
                    "gold",
                    "active",
                    Decimal("250.00"),
                    "USD",
                ),
                Customer(
                    "CUST-00002",
                    "Grace Hopper",
                    "grace@example.com",
                    "standard",
                    "active",
                    Decimal("40.00"),
                    "USD",
                ),
                Customer(
                    "CUST-00003",
                    "Alan Turing",
                    "alan@example.com",
                    "standard",
                    "suspended",
                    Decimal("120.00"),
                    "GBP",
                ),
                Customer(
                    "CUST-00004",
                    "Edsger Dijkstra",
                    "edsger@example.com",
                    "gold",
                    "closed",
                    Decimal("0.00"),
                    "EUR",
                ),
                Customer(
                    "CUST-00005",
                    "Barbara Liskov",
                    "barbara@example.com",
                    "platinum",
                    "active",
                    Decimal("10000.00"),
                    "EUR",
                ),
            ]
        )

    def get(self, customer_id: str) -> Customer | None:
        return self._customers.get(customer_id)

    def save(self, customer: Customer) -> None:
        self._customers[customer.customer_id] = customer


def _default_refund_id() -> str:
    return f"RF-{uuid4().hex[:12].upper()}"


class CustomerService:
    def __init__(
        self,
        repository: CustomerRepository,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        refund_id_factory: Callable[[], str] = _default_refund_id,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._refund_id_factory = refund_id_factory
        self._refunds: list[RefundReceipt] = []

    @property
    def refunds(self) -> tuple[RefundReceipt, ...]:
        """Refunds accepted during the lifetime of this service (audit trail)."""
        return tuple(self._refunds)

    def get_customer_record(self, customer_id: str) -> CustomerRecord:
        customer = self._require_customer(customer_id)
        return CustomerRecord(
            customer_id=customer.customer_id,
            full_name=customer.full_name,
            email=customer.email,
            tier=customer.tier,
            status=customer.status,
            refundable_balance=float(customer.refundable_balance),
            currency=customer.currency,
        )

    def trigger_refund(self, customer_id: str, amount: float, reason: str) -> RefundReceipt:
        customer = self._require_customer(customer_id)
        if customer.status != "active":
            raise CustomerNotEligibleError(
                f"Customer {customer_id} is {customer.status}; refunds require an active account"
            )
        requested = Decimal(str(amount)).quantize(_CENT)
        if requested > customer.refundable_balance:
            raise RefundExceedsBalanceError(
                f"Requested {requested} {customer.currency} exceeds the refundable balance of "
                f"{customer.refundable_balance} {customer.currency}"
            )

        updated = replace(customer, refundable_balance=customer.refundable_balance - requested)
        self._repository.save(updated)
        receipt = RefundReceipt(
            refund_id=self._refund_id_factory(),
            customer_id=customer_id,
            amount=float(requested),
            currency=customer.currency,
            reason=reason,
            status="accepted",
            remaining_balance=float(updated.refundable_balance),
            created_at=self._clock(),
        )
        self._refunds.append(receipt)
        logger.info(
            "refund accepted",
            extra={
                "event": "refund.accepted",
                "refund_id": receipt.refund_id,
                "customer_id": customer_id,
                "amount": receipt.amount,
                "currency": receipt.currency,
            },
        )
        return receipt

    def _require_customer(self, customer_id: str) -> Customer:
        customer = self._repository.get(customer_id)
        if customer is None:
            raise CustomerNotFoundError(f"Customer {customer_id} does not exist")
        return customer
