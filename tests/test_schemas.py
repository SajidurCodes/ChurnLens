import pytest
from pydantic import ValidationError

from churnlens.features import MODEL_FEATURES
from churnlens.schemas import PredictionRequest


@pytest.fixture
def valid_payload():
    return {
        "call_failure": 2,
        "subscription_length": 24,
        "seconds_of_use": 1200,
        "frequency_of_use": 40,
        "frequency_of_sms": 15,
        "distinct_called_numbers": 10,
        "charge_amount": 10,
        "age_group": 3,
        "complains": 0,
        "tariff_plan": 1,
    }


def test_valid_payload_preserves_values(valid_payload):
    request = PredictionRequest.model_validate(valid_payload)
    assert request.model_dump() == valid_payload


@pytest.mark.parametrize(
    "invalid",
    ["10", 10.0, True, None, float("nan"), float("inf")],
)
def test_integer_field_rejects_coercion(valid_payload, invalid):
    valid_payload["seconds_of_use"] = invalid

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(valid_payload)


@pytest.mark.parametrize("invalid", [-1, 2_147_483_648])
def test_numeric_bounds(valid_payload, invalid):
    valid_payload["frequency_of_sms"] = invalid

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(valid_payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("complains", 2),
        ("tariff_plan", 0),
        ("age_group", 6),
        ("charge_amount", 11),
    ],
)
def test_invalid_category_rejected(valid_payload, field, invalid):
    valid_payload[field] = invalid

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(valid_payload)


def test_missing_field_rejected(valid_payload):
    del valid_payload["subscription_length"]

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(valid_payload)


@pytest.mark.parametrize("extra_field", ["churn", "status", "customer_value"])
def test_extra_field_rejected(valid_payload, extra_field):
    valid_payload[extra_field] = 1

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(valid_payload)


def test_request_matches_training_feature_contract():
    assert set(PredictionRequest.model_fields) == set(MODEL_FEATURES)