"""Schema validation: every accepted and rejected shape is pinned here."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from customer_mcp_server.schemas import (
    CUSTOMER_ID_PATTERN,
    MAX_REFUND_AMOUNT,
    GetCustomerRecordInput,
    TriggerRefundInput,
)

VALID_REFUND: dict[str, Any] = {
    "customer_id": "CUST-00001",
    "amount": 49.99,
    "reason": "Customer returned the item within the window.",
}


def _error_types(exc_info: pytest.ExceptionInfo[ValidationError]) -> set[tuple[str, str]]:
    return {(".".join(str(p) for p in err["loc"]), err["type"]) for err in exc_info.value.errors()}


@pytest.mark.parametrize("customer_id", ["CUST-00001", "CUST-99999", "CUST-00000"])
def test_customer_id_accepts_canonical_format(customer_id: str) -> None:
    assert GetCustomerRecordInput(customer_id=customer_id).customer_id == customer_id


@pytest.mark.parametrize(
    "customer_id",
    [
        "cust-00001",  # lower case prefix
        "CUST-0001",  # four digits
        "CUST-000001",  # six digits
        "CUST-ABCDE",  # letters instead of digits
        "CUST00001",  # missing hyphen
        " CUST-00001",  # leading whitespace
        "CUST-00001\n",  # trailing newline
        "CUST-\u0660\u0661\u0662\u0663\u0664",  # Arabic-Indic digits
        "CUST-\uff10\uff11\uff12\uff13\uff14",  # fullwidth digits
        "CUST-\u00b2\u00b3\u00b9\u00bc\u00bd",  # superscripts and fractions
        "CUST-00001; DROP TABLE customers",
        "",
    ],
)
def test_customer_id_rejects_malformed_values(customer_id: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        GetCustomerRecordInput(customer_id=customer_id)
    assert ("customer_id", "string_pattern_mismatch") in _error_types(exc_info)


@pytest.mark.parametrize("value", [12345, None, ["CUST-00001"], {"id": "CUST-00001"}, True])
def test_customer_id_rejects_non_strings(value: Any) -> None:
    with pytest.raises(ValidationError) as exc_info:
        GetCustomerRecordInput.model_validate({"customer_id": value})
    assert any(path == "customer_id" for path, _ in _error_types(exc_info))


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        GetCustomerRecordInput.model_validate({"customer_id": "CUST-00001", "verbose": True})
    assert ("verbose", "extra_forbidden") in _error_types(exc_info)


def test_missing_fields_are_reported_individually() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TriggerRefundInput.model_validate({})
    assert _error_types(exc_info) == {
        ("customer_id", "missing"),
        ("amount", "missing"),
        ("reason", "missing"),
    }


@pytest.mark.parametrize("amount", [0.01, 1, 100, 49.99, 1000000, MAX_REFUND_AMOUNT])
def test_refund_amount_accepts_positive_numbers(amount: float) -> None:
    model = TriggerRefundInput.model_validate({**VALID_REFUND, "amount": amount})
    assert model.amount == pytest.approx(float(amount))
    assert isinstance(model.amount, float)


@pytest.mark.parametrize(
    ("amount", "expected_type"),
    [
        (0, "greater_than"),
        (-0.01, "greater_than"),
        (-100, "greater_than"),
        (MAX_REFUND_AMOUNT + 0.01, "less_than_equal"),
        (math.nan, "finite_number"),
        (math.inf, "finite_number"),
        (-math.inf, "finite_number"),
        ("49.99", "float_type"),  # numeric strings are not coerced in strict mode
        (True, "float_type"),  # booleans are not numbers
        (None, "float_type"),
        ([49.99], "float_type"),
        (10.005, "value_error"),  # more than two decimal places
        (1e-05, "value_error"),
    ],
)
def test_refund_amount_rejects_invalid_values(amount: Any, expected_type: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TriggerRefundInput.model_validate({**VALID_REFUND, "amount": amount})
    assert ("amount", expected_type) in _error_types(exc_info)


def test_refund_amount_rejects_nan_in_json_payloads() -> None:
    payload = '{"customer_id": "CUST-00001", "amount": NaN, "reason": "ten characters!"}'
    with pytest.raises(ValidationError):
        TriggerRefundInput.model_validate_json(payload)


@pytest.mark.parametrize(
    "reason",
    [
        "short",
        "123456789",  # nine characters
        "          ",  # ten spaces
        "   seven   ",  # ten characters but only five after trimming
        "",
    ],
)
def test_reason_rejects_too_short_values(reason: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason})
    assert any(path == "reason" for path, _ in _error_types(exc_info))


def test_reason_trimming_is_applied_before_the_length_check() -> None:
    with pytest.raises(ValidationError):
        TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "  exactly10  "})
    model = TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "  exactly ten  "})
    assert model.reason == "exactly ten"


def test_reason_rejects_control_characters_and_over_long_text() -> None:
    with pytest.raises(ValidationError):
        TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "bad\x00reason here"})
    with pytest.raises(ValidationError):
        TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "x" * 501})


def test_reason_rejects_non_strings() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TriggerRefundInput.model_validate({**VALID_REFUND, "reason": 1234567890})
    assert ("reason", "string_type") in _error_types(exc_info)


def test_json_schema_advertises_the_constraints() -> None:
    schema = TriggerRefundInput.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"customer_id", "amount", "reason"}
    assert (
        schema["properties"]["customer_id"]["pattern"] == CUSTOMER_ID_PATTERN == r"^CUST-[0-9]{5}$"
    )
    assert schema["properties"]["amount"]["exclusiveMinimum"] == 0
    assert schema["properties"]["reason"]["minLength"] == 10


@pytest.mark.parametrize(
    "reason",
    [
        "Refund\u00a0for\u00a0broken\u00a0item",  # non-breaking spaces
        "Soft\u00adhyphen in the reason",  # soft hyphen
        "Family emoji \U0001f468\u200d\U0001f469\u200d\U0001f467 was unhappy",  # ZWJ
        "\u200f\u0645\u0631\u062a\u062c\u0639 \u0645\u0643\u0633\u0648\u0631 broken",  # RTL mark
        "Damaged item\r\nreturned by customer",  # CRLF newlines
        "x" * 495 + " " * 10,  # 500 characters only because of trailing padding
    ],
)
def test_reason_accepts_ordinary_international_text(reason: str) -> None:
    model = TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason})
    assert model.reason == reason.strip()


@pytest.mark.parametrize(
    "reason",
    [
        "\x1cDamaged item returned\x1c",  # C0 controls hiding in trimmed whitespace
        "Damaged\x1bitem returned",  # escape character inside
        "\u0301" * 12,  # combining marks without a base character
        "!!!!!!!!!!!!!!",  # punctuation only
    ],
)
def test_reason_rejects_controls_and_meaningless_text(reason: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason})
    assert ("reason", "value_error") in _error_types(exc_info)
