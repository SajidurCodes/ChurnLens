import json
import shutil

import joblib
import pandas as pd
import pytest
from pydantic import ValidationError

from churnlens.features import MODEL_FEATURES
from churnlens.inference import PROJECT_ROOT, PredictionService


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


@pytest.fixture(scope="module")
def service():
    return PredictionService()


@pytest.fixture
def copied_bundle(tmp_path):
    release_relative = "reports/frozen_model_manifest.json"
    release_source = PROJECT_ROOT / release_relative
    release = json.loads(release_source.read_text(encoding="utf-8"))

    relative_paths = [
        release_relative,
        *(record["path"] for record in release["files"].values()),
    ]

    for relative in relative_paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, destination)

    return tmp_path


def test_serving_matches_offline_prediction(service, payload):
    model = joblib.load(
        PROJECT_ROOT / "artifacts" / "frozen" / "model.joblib"
    )
    frame = pd.DataFrame([payload], columns=list(MODEL_FEATURES))
    expected = float(model.predict_proba(frame)[0, 1])

    result = service.predict(payload)

    assert result["churn_probability"] == pytest.approx(
        expected, abs=1e-12, rel=0
    )
    assert result["threshold"] == 0.2216680816787302
    assert result["flagged_for_review"] == (
        expected >= result["threshold"]
    )
    assert result["model_version"] == "churnlens-v0.1"


def test_json_key_order_does_not_change_prediction(service, payload):
    reversed_payload = dict(reversed(list(payload.items())))

    first = service.predict(payload)
    second = service.predict(reversed_payload)

    assert second["churn_probability"] == pytest.approx(
        first["churn_probability"], abs=1e-12, rel=0
    )
    assert second["flagged_for_review"] == first["flagged_for_review"]


def test_service_rejects_invalid_input(service, payload):
    payload["seconds_of_use"] = "1200"

    with pytest.raises(ValidationError):
        service.predict(payload)


def test_model_corruption_rejected_before_deserialization(
    copied_bundle, monkeypatch
):
    model_path = copied_bundle / "artifacts" / "frozen" / "model.joblib"
    model_path.write_bytes(b"corrupted model")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Deserialization must not run after checksum failure.")

    monkeypatch.setattr("churnlens.inference.joblib.load", unexpected_load)

    with pytest.raises(ValueError, match="checksum mismatch: model.joblib"):
        PredictionService(copied_bundle)


def test_manifest_modification_rejected(copied_bundle):
    path = copied_bundle / "reports" / "frozen_model_manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        PredictionService(copied_bundle)


def test_missing_model_rejected(copied_bundle):
    path = copied_bundle / "artifacts" / "frozen" / "model.joblib"
    path.unlink()

    with pytest.raises(FileNotFoundError):
        PredictionService(copied_bundle)