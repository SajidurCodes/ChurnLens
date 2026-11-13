import json

import pytest
from fastapi.testclient import TestClient

from churnlens.api import create_app
from churnlens.inference import PredictionService


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def payload():
    return {
        "call_failure": 2,
        "subscription_length": 24,
        "seconds_of_use": 1200,
        "frequency_of_use": 40,
        "frequency_of_sms": 15,
        "distinct_called_numbers": 10,
        "charge_amount": 3,
        "age_group": 3,
        "complains": 0,
        "tariff_plan": 1,
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_version": "churnlens-v0.1",
    }


def test_api_matches_direct_inference(client, payload):
    expected = PredictionService().predict(payload)
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    actual = response.json()
    assert actual["churn_probability"] == pytest.approx(
        expected["churn_probability"], abs=1e-12, rel=0
    )
    assert actual["threshold"] == expected["threshold"]
    assert actual["flagged_for_review"] == expected["flagged_for_review"]
    assert actual["model_version"] == expected["model_version"]


@pytest.mark.parametrize("invalid", ["1200", 1200.0, True, -1])
def test_invalid_numeric_input_returns_422(client, payload, invalid):
    payload["seconds_of_use"] = invalid
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "seconds_of_use"]
        for error in response.json()["detail"]
    )


def test_missing_field_returns_422(client, payload):
    del payload["tariff_plan"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_extra_field_returns_422(client, payload):
    payload["churn"] = 1

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_nonfinite_input_returns_json_error(client, payload):
    payload["seconds_of_use"] = float("nan")

    # Deliberately send a nonstandard JSON numeric value to test rejection.
    response = client.post(
        "/predict",
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert all("input" not in error for error in response.json()["detail"])


def test_malformed_json_returns_422(client):
    response = client.post(
        "/predict",
        content='{"seconds_of_use":',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_missing_bundle_prevents_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(root=tmp_path)):
            pytest.fail("Startup should have failed.")