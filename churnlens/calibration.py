"""Grouped folds and sigmoid calibration for training data only."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

from churnlens.features import MODEL_FEATURES


def calibration_folds(
    X: pd.DataFrame,
    y: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build positional folds; X and y must have matching row indices."""
    if not X.index.equals(y.index):
        raise ValueError("X and y indices must match.")

    missing = sorted(set(MODEL_FEATURES) - set(X.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    features = X[list(MODEL_FEATURES)]
    labels = y.to_numpy()

    if features.empty or features.isna().any().any() or y.isna().any():
        raise ValueError("Training data is empty or contains missing values.")

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain non-finite values.")

    if set(labels) != {0, 1}:
        raise ValueError("Training labels must contain classes 0 and 1.")

    groups = features.groupby(
        list(MODEL_FEATURES),
        sort=False,
        dropna=False,
    ).ngroup().to_numpy()

    if len(np.unique(groups)) < 3:
        raise ValueError("At least three profile groups are required.")

    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )
    folds = list(splitter.split(features, labels, groups=groups))
    visits = np.zeros(len(features), dtype=int)

    for fit_ids, heldout_ids in folds:
        if set(groups[fit_ids]) & set(groups[heldout_ids]):
            raise ValueError("Profile overlap in calibration folds.")

        for ids in (fit_ids, heldout_ids):
            if set(labels[ids]) != {0, 1}:
                raise ValueError("A calibration partition lacks a class.")

        visits[heldout_ids] += 1

    if not (visits == 1).all():
        raise ValueError("Each training row needs one held-out prediction.")

    return folds


def build_calibrated(
    estimator: BaseEstimator,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> CalibratedClassifierCV:
    """Wrap an unfitted pipeline using folds for the same training rows."""
    return CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=folds,
        ensemble=False,
        n_jobs=1,
    )