import pandas as pd
import pytest

from churnlens.features import MODEL_FEATURES
from churnlens.splitting import make_manifest


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    rows = []
    for profile in range(50):
        base = {feature: 1 for feature in MODEL_FEATURES}
        base["seconds_of_use"] = profile * 100

        # Same model inputs, different labels and excluded fields.
        rows.append({**base, "churn": 0, "status": 1})
        rows.append({**base, "churn": 1, "status": 2})

    return pd.DataFrame(rows)


def test_every_row_appears_once(sample_frame):
    manifest = make_manifest(sample_frame)

    assert len(manifest) == len(sample_frame)
    assert manifest["row_id"].is_unique
    assert set(manifest["row_id"]) == set(range(len(sample_frame)))
    assert set(manifest["split"]) == {"train", "validation", "test"}


def test_matching_inputs_never_cross_splits(sample_frame):
    manifest = make_manifest(sample_frame)

    # Check actual input profiles independently of generated group IDs.
    joined = sample_frame[list(MODEL_FEATURES)].copy()
    joined["split"] = manifest["split"]

    assert joined.groupby(list(MODEL_FEATURES))["split"].nunique().max() == 1
    assert manifest["group_id"].nunique() == 50


def test_reproducible_and_does_not_modify_input(sample_frame):
    original = sample_frame.copy(deep=True)

    first = make_manifest(sample_frame)
    second = make_manifest(sample_frame)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(sample_frame, original)


def test_both_classes_in_each_split(sample_frame):
    manifest = make_manifest(sample_frame)

    for name in ("train", "validation", "test"):
        row_ids = manifest.loc[manifest["split"] == name, "row_id"]
        assert set(sample_frame.iloc[row_ids]["churn"]) == {0, 1}


def test_missing_feature_rejected(sample_frame):
    with pytest.raises(ValueError, match="Missing required"):
        make_manifest(sample_frame.drop(columns="complains"))


def test_single_class_rejected(sample_frame):
    sample_frame["churn"] = 0

    with pytest.raises(ValueError, match="exactly classes"):
        make_manifest(sample_frame)


def test_too_few_profiles_rejected(sample_frame):
    with pytest.raises(ValueError, match="five distinct"):
        make_manifest(sample_frame.iloc[:8])