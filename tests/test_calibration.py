import joblib
import numpy as np
import pandas as pd
import pytest

from churnlens.calibration import build_calibrated, calibration_folds
from churnlens.features import MODEL_FEATURES
from churnlens.models import build_logistic


@pytest.fixture
def training_data():
    rows = []
    labels = []

    for profile in range(30):
        row = {name: 1 for name in MODEL_FEATURES}
        row["seconds_of_use"] = 100 + profile * 20
        row["complains"] = profile % 2
        row["tariff_plan"] = 1 + profile % 2

        # Matching inputs with conflicting labels must stay together.
        rows.extend([row.copy(), row.copy()])
        labels.extend([0, 1])

    return pd.DataFrame(rows), pd.Series(labels, name="churn")


def test_calibration_profiles_stay_together(training_data):
    X, y = training_data
    folds = calibration_folds(X, y)

    for fit_ids, heldout_ids in folds:
        fit_profiles = set(
            X.iloc[fit_ids][list(MODEL_FEATURES)].itertuples(
                index=False, name=None
            )
        )
        heldout_profiles = set(
            X.iloc[heldout_ids][list(MODEL_FEATURES)].itertuples(
                index=False, name=None
            )
        )
        assert fit_profiles.isdisjoint(heldout_profiles)


def test_each_row_gets_one_heldout_prediction(training_data):
    X, y = training_data
    folds = calibration_folds(X, y)

    heldout = np.concatenate([ids for _, ids in folds])
    np.testing.assert_array_equal(np.sort(heldout), np.arange(len(X)))


def test_misaligned_labels_rejected(training_data):
    X, y = training_data

    with pytest.raises(ValueError, match="indices must match"):
        calibration_folds(X, y.iloc[::-1])


def test_calibrated_pipeline_round_trip(training_data, tmp_path):
    X, y = training_data
    model = build_calibrated(
        build_logistic(),
        calibration_folds(X, y),
    )
    model.fit(X, y)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    path = tmp_path / "calibrated.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)

    np.testing.assert_allclose(
        restored.predict_proba(X),
        probabilities,
    )