import numpy as np
import pytest

from churnlens.thresholds import select_threshold


def test_maximizes_churners_caught_within_capacity():
    result = select_threshold(
        [1, 0, 1, 0, 1],
        [0.9, 0.8, 0.7, 0.6, 0.1],
        max_flag_fraction=0.6,
    )

    assert result["threshold"] == pytest.approx(0.7)
    assert result["flagged_rows"] == 3
    assert result["confusion_matrix"] == [[1, 1], [1, 2]]


def test_prefers_fewer_flags_when_recall_is_equal():
    result = select_threshold(
        [1, 0, 0, 1],
        [0.9, 0.8, 0.7, 0.1],
        max_flag_fraction=0.75,
    )

    assert result["threshold"] == pytest.approx(0.9)
    assert result["flagged_rows"] == 1


def test_tied_scores_are_not_split():
    result = select_threshold(
        [1, 0, 1, 0],
        [0.8, 0.8, 0.8, 0.1],
        max_flag_fraction=0.5,
    )

    assert result["threshold"] is None
    assert result["flagged_rows"] == 0


def test_capacity_rounds_down():
    result = select_threshold(
        np.ones(636, dtype=int),
        np.linspace(0, 1, 636),
    )

    assert result["capacity_rows"] == 127
    assert result["flagged_rows"] == 127


@pytest.mark.parametrize("invalid", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_probability_rejected(invalid):
    with pytest.raises(ValueError, match="Probabilities"):
        select_threshold([1], [invalid])