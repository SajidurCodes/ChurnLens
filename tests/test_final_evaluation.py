import pytest

from scripts.evaluate_final_test import decision_metrics


def test_threshold_is_inclusive():
    result = decision_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.2, 0.9],
        threshold=0.2,
    )

    assert result["flagged_rows"] == 3
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2 / 3)


def test_null_threshold_flags_nobody():
    result = decision_metrics(
        [0, 1],
        [0.1, 1.0],
        threshold=None,
    )

    assert result["flagged_rows"] == 0
    assert result["confusion_matrix"] == [[1, 0], [1, 0]]


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError, match="Invalid threshold"):
        decision_metrics([0, 1], [0.1, 0.9], threshold=-0.1)