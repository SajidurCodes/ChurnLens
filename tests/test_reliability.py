import numpy as np
import pytest

from scripts.compare_calibration import reliability_bins


def test_boundaries_and_empty_bins():
    bins = reliability_bins(
        [0, 1, 1],
        [0.0, 0.2, 1.0],
    )

    assert [row["count"] for row in bins] == [1, 1, 0, 0, 1]
    assert bins[0]["mean_probability"] == 0.0
    assert bins[1]["mean_probability"] == 0.2
    assert bins[4]["mean_probability"] == 1.0
    assert bins[2]["mean_probability"] is None
    assert bins[2]["observed_churn_rate"] is None


def test_bin_statistics():
    bins = reliability_bins(
        [0, 1, 1],
        [0.05, 0.10, 0.15],
    )

    assert bins[0]["count"] == 3
    assert bins[0]["mean_probability"] == pytest.approx(0.10)
    assert bins[0]["observed_churn_rate"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("invalid", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_probability_rejected(invalid):
    with pytest.raises(ValueError, match="Probabilities"):
        reliability_bins([0], [invalid])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="matching vectors"):
        reliability_bins([0, 1], [0.5])