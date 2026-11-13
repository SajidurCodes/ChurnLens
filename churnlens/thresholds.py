"""Select a validation threshold under an explicit review capacity."""

import math

import numpy as np


def select_threshold(
    y,
    probabilities,
    max_flag_fraction: float = 0.20,
) -> dict:
    labels = np.asarray(y)
    p = np.asarray(probabilities, dtype=float)

    if labels.ndim != 1 or p.ndim != 1 or len(labels) != len(p):
        raise ValueError("Labels and probabilities must be matching vectors.")
    if len(p) == 0:
        raise ValueError("Inputs cannot be empty.")
    if not np.isin(labels, [0, 1]).all() or not (labels == 1).any():
        raise ValueError("Labels must be binary and include churners.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    if not np.isfinite(max_flag_fraction) or not 0 <= max_flag_fraction <= 1:
        raise ValueError("Capacity fraction must be within [0, 1].")

    capacity = math.floor(len(p) * max_flag_fraction)

    # None means flag nobody, including when tied scores exceed capacity.
    best_threshold = None
    best_tp = 0
    best_flagged = 0

    for threshold in np.unique(p):
        flagged = p >= threshold
        count = int(flagged.sum())
        if count > capacity:
            continue

        tp = int(((labels == 1) & flagged).sum())
        if tp > best_tp or (tp == best_tp and count < best_flagged):
            best_threshold = float(threshold)
            best_tp = tp
            best_flagged = count

    positives = int((labels == 1).sum())
    negatives = len(labels) - positives
    fp = best_flagged - best_tp
    fn = positives - best_tp
    tn = negatives - fp

    return {
        "threshold": best_threshold,
        "comparison": ">=",
        "null_threshold_means": "flag nobody",
        "max_flag_fraction": float(max_flag_fraction),
        "capacity_rows": capacity,
        "flagged_rows": best_flagged,
        "flagged_fraction": best_flagged / len(p),
        "precision": best_tp / best_flagged if best_flagged else 0.0,
        "recall": best_tp / positives,
        "f1": (
            2 * best_tp / (2 * best_tp + fp + fn)
            if 2 * best_tp + fp + fn
            else 0.0
        ),
        "confusion_matrix": [[tn, fp], [fn, best_tp]],
    }