import joblib
import numpy as np
import pandas as pd

from churnlens.features import MODEL_FEATURES
from churnlens.models import build_forest


def training_example():
    rows = []
    for i in range(40):
        row = {name: 1 for name in MODEL_FEATURES}
        row["seconds_of_use"] = 100 + i * 20
        row["complains"] = i % 2
        row["tariff_plan"] = 1 + i % 2
        rows.append(row)

    return pd.DataFrame(rows), np.array([i % 2 for i in range(40)])


def test_excluded_columns_do_not_affect_predictions():
    X, y = training_example()
    model = build_forest().fit(X, y)
    expected = model.predict_proba(X)

    changed = X.copy()
    changed["status"] = 2
    changed["customer_value"] = 99999
    changed["age"] = 99
    changed["churn"] = 1

    np.testing.assert_allclose(
        model.predict_proba(changed),
        expected,
    )


def test_forest_round_trip(tmp_path):
    X, y = training_example()
    model = build_forest().fit(X, y)

    path = tmp_path / "forest.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    probabilities = restored.predict_proba(X)

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities, model.predict_proba(X))