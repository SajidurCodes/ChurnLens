"""Reproducible splits that keep identical model inputs together."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from churnlens.features import MODEL_FEATURES, TARGET


SPLIT_SEED = 42


def make_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Return positional row IDs, profile groups, and split assignments.

    Input must have normalized column names and numeric model features.
    Row IDs refer to the unchanged source CSV's zero-based data-row order.
    """
    required = [*MODEL_FEATURES, TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = df.reset_index(drop=True)
    values = frame[required]

    if frame.empty or values.isna().any().any():
        raise ValueError("Input is empty or contains missing values.")

    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Input contains non-finite values.")

    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain exactly classes 0 and 1.")

    # Neither the target nor excluded features participate in grouping.
    groups = frame.groupby(
        list(MODEL_FEATURES), sort=False, dropna=False
    ).ngroup()

    if groups.nunique() < 5:
        raise ValueError("At least five distinct input profiles are required.")

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPLIT_SEED,
    )

    assignments = np.full(len(frame), "", dtype=object)
    visits = np.zeros(len(frame), dtype=int)

    for fold, (_, held_out) in enumerate(
        splitter.split(
            frame[list(MODEL_FEATURES)],
            frame[TARGET],
            groups=groups,
        )
    ):
        name = "test" if fold == 0 else "validation" if fold == 1 else "train"
        assignments[held_out] = name
        visits[held_out] += 1

    if not (visits == 1).all():
        raise ValueError("Each row must be assigned exactly once.")

    manifest = pd.DataFrame(
        {
            "row_id": np.arange(len(frame)),
            "group_id": groups.to_numpy(),
            "split": assignments,
        }
    )

    if manifest.groupby("group_id")["split"].nunique().max() != 1:
        raise ValueError("An input profile crosses split boundaries.")

    for name in ("train", "validation", "test"):
        labels = frame.loc[assignments == name, TARGET]
        if set(labels.unique()) != {0, 1}:
            raise ValueError(f"{name} does not contain both target classes.")

    return manifest