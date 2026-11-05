"""Domain behaviour of the customer service."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from customer_mcp_server.errors import (
    CustomerNotEligibleError,
    CustomerNotFoundError,
    RefundExceedsBalanceError,
)
from customer_mcp_server.service import CustomerService, InMemoryCustomerRepository


def test_get_customer_record_returns_projection(service: CustomerService) -> None:
    record = service.get_customer_record("CUST-00001")
    assert record.customer_id == "CUST-00001"
    assert record.refundable_balance == 250.0
    assert record.currency == "USD"


def test_get_customer_record_unknown_customer(service: CustomerService) -> None:
    with pytest.raises(CustomerNotFoundError):
        service.get_customer_record("CUST-99999")


def test_refund_reduces_balance_and_is_recorded() -> None:
    fixed_now = datetime(2026, 1, 1, tzinfo=UTC)
    service = CustomerService(
        InMemoryCustomerRepository.with_sample_data(),
        clock=lambda: fixed_now,
        refund_id_factory=lambda: "RF-TEST",
    )
    receipt = service.trigger_refund("CUST-00001", 25.5, "Damaged item returned")
    assert receipt.refund_id == "RF-TEST"
    assert receipt.amount == 25.5
    assert receipt.remaining_balance == 224.5
    assert receipt.created_at == fixed_now
    assert service.get_customer_record("CUST-00001").refundable_balance == 224.5
    assert service.refunds == (receipt,)


def test_refund_uses_exact_decimal_arithmetic(service: CustomerService) -> None:
    # 0.1 + 0.2 style float drift must not leak into balances.
    for _ in range(10):
        service.trigger_refund("CUST-00001", 0.1, "Ten small refunds")
    assert service.get_customer_record("CUST-00001").refundable_balance == 249.0


def test_refund_rejects_amount_above_balance(service: CustomerService) -> None:
    with pytest.raises(RefundExceedsBalanceError):
        service.trigger_refund("CUST-00002", 40.01, "One cent too many")
    # An exact match of the balance is allowed.
    receipt = service.trigger_refund("CUST-00002", 40.0, "Full balance refund")
    assert receipt.remaining_balance == 0.0


@pytest.mark.parametrize("customer_id", ["CUST-00003", "CUST-00004"])
def test_refund_requires_active_customer(service: CustomerService, customer_id: str) -> None:
    with pytest.raises(CustomerNotEligibleError):
        service.trigger_refund(customer_id, 1.0, "Account state check")


def test_refund_unknown_customer(service: CustomerService) -> None:
    with pytest.raises(CustomerNotFoundError):
        service.trigger_refund("CUST-99999", 1.0, "Nobody lives here")
