import pandas as pd
import pytest

from scripts.audit_data import audit_frame


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    row = {
        "Call  Failure": 2,
        "Complains": 0,
        "Subscription  Length": 12,
        "Charge  Amount": 3,
        "Seconds of Use": 1200,
        "Frequency of use": 40,
        "Frequency of SMS": 15,
        "Distinct Called Numbers": 10,
        "Age Group": 3,
        "Tariff Plan": 1,
        "Status": 1,
        "Age": 35,
        "Customer Value": 150.5,
        "Churn": 0,
    }
    # One repeated row and one matching profile with a different label.
    return pd.DataFrame([row, row.copy(), {**row, "Churn": 1}])


def test_duplicates_and_conflicting_labels(sample_frame):
    original = sample_frame.copy(deep=True)
    report = audit_frame(sample_frame)

    assert report["rows"] == 3
    assert report["class_counts"] == {"0": 2, "1": 1}
    assert report["duplicate_rows_beyond_first"] == 1
    assert report["duplicate_input_profiles_beyond_first"] == 2
    assert report["conflicting_label_groups"] == 1
    pd.testing.assert_frame_equal(sample_frame, original)


def test_missing_column_is_rejected(sample_frame):
    with pytest.raises(ValueError, match="Schema mismatch"):
        audit_frame(sample_frame.drop(columns="Churn"))


def test_invalid_target_is_rejected(sample_frame):
    sample_frame.loc[0, "Churn"] = 2
    with pytest.raises(ValueError, match="Invalid category in churn"):
        audit_frame(sample_frame)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1, 1.5])
def test_invalid_call_count_is_rejected(sample_frame, bad_value):
    sample_frame["Call  Failure"] = sample_frame["Call  Failure"].astype(float)
    sample_frame.loc[0, "Call  Failure"] = bad_value

    with pytest.raises(ValueError):
        audit_frame(sample_frame)


def test_normalized_name_collision_is_rejected(sample_frame):
    sample_frame["call_failure"] = 0
    with pytest.raises(ValueError, match="collide"):
        audit_frame(sample_frame)


def test_charge_band_discrepancy_is_reported(sample_frame):
    sample_frame.loc[0, "Charge  Amount"] = 10
    report = audit_frame(sample_frame)

    assert report["ranges"]["charge_amount"]["max"] == 10
    assert len(report["warnings"]) == 1