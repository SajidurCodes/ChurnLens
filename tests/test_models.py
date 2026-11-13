import joblib
import numpy as np
import pandas as pd

from churnlens.features import (
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
)
from churnlens.models import build_logistic


def training_example():
    rows = []
    for i in range(20):
        row = {name: 1 for name in MODEL_FEATURES}
        row["seconds_of_use"] = 100 + i * 20
        row["complains"] = i % 2
        row["tariff_plan"] = 1 + i % 2
        rows.append(row)

    return pd.DataFrame(rows), np.array([i % 2 for i in range(20)])


def test_scaler_uses_training_statistics():
    X, y = training_example()
    pipeline = build_logistic().fit(X, y)

    scaler = pipeline.named_steps[
        "preprocessing"
    ].named_transformers_["numeric"]

    expected = X[list(NUMERIC_FEATURES + ORDINAL_FEATURES)].mean()
    np.testing.assert_allclose(scaler.mean_, expected.to_numpy())


def test_prediction_does_not_refit_preprocessing():
    X, y = training_example()
    pipeline = build_logistic().fit(X, y)

    scaler = pipeline.named_steps[
        "preprocessing"
    ].named_transformers_["numeric"]
    before = scaler.mean_.copy()

    validation = X.iloc[:3].copy()
    validation["seconds_of_use"] = 100000
    probabilities = pipeline.predict_proba(validation)

    np.testing.assert_array_equal(scaler.mean_, before)
    assert probabilities.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_saved_pipeline_preserves_predictions(tmp_path):
    X, y = training_example()
    pipeline = build_logistic().fit(X, y)

    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)

    np.testing.assert_allclose(
        restored.predict_proba(X),
        pipeline.predict_proba(X),
    )