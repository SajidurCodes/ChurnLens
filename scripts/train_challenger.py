"""Compare fixed model configurations using training-only grouped CV."""

import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from churnlens.features import MODEL_FEATURES, TARGET
from churnlens.models import build_forest, build_logistic
from scripts.audit_data import (
    DATA_PATH,
    EXPECTED_SHA256,
    ROOT,
    normalize_column,
)
from scripts.train_baselines import EXPECTED_MANIFEST_SHA256, evaluate


def main() -> None:
    source_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if source_hash != EXPECTED_SHA256:
        raise ValueError("Source checksum mismatch.")

    manifest_path = ROOT / "data" / "splits" / "manifest.csv"
    manifest_hash = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Manifest checksum mismatch.")

    frame = pd.read_csv(DATA_PATH).rename(columns=normalize_column)
    manifest = pd.read_csv(manifest_path)

    train_manifest = manifest.loc[
        manifest["split"] == "train"
    ].reset_index(drop=True)
    train = frame.iloc[train_manifest["row_id"]].reset_index(drop=True)

    validation_ids = manifest.loc[
        manifest["split"] == "validation", "row_id"
    ]
    validation = frame.iloc[validation_ids].reset_index(drop=True)

    X_train = train[list(MODEL_FEATURES)]
    y_train = train[TARGET]
    groups = train_manifest["group_id"]

    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )
    folds = list(splitter.split(X_train, y_train, groups=groups))

    fold_summary = []
    for fit_ids, holdout_ids in folds:
        overlap = set(groups.iloc[fit_ids]) & set(groups.iloc[holdout_ids])
        if overlap:
            raise ValueError("Profile overlap inside cross-validation.")

        for ids in (fit_ids, holdout_ids):
            if set(y_train.iloc[ids].unique()) != {0, 1}:
                raise ValueError("A cross-validation partition lacks a class.")

        fold_summary.append(
            {
                "fit_rows": len(fit_ids),
                "holdout_rows": len(holdout_ids),
                "overlapping_groups": len(overlap),
            }
        )

    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "neg_brier_score": "neg_brier_score",
        "neg_log_loss": "neg_log_loss",
    }

    cv_results = {}
    for name, model in {
        "logistic_regression": build_logistic(),
        "random_forest": build_forest(),
    }.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=folds,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )

        metrics = {}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            # sklearn negates losses for scoring; restore readable losses.
            if metric.startswith("neg_"):
                values = -values
            output_name = metric.removeprefix("neg_")
            metrics[output_name] = {
                "fold_values": values.tolist(),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        cv_results[name] = metrics

    forest = build_forest()
    forest.fit(X_train, y_train)

    positive_column = list(forest.classes_).index(1)
    probabilities = forest.predict_proba(
        validation[list(MODEL_FEATURES)]
    )[:, positive_column]

    report = {
        "source_sha256": source_hash,
        "manifest_sha256": manifest_hash,
        "configuration": forest.named_steps["model"].get_params(),
        "cv_scope": "Training rows only; same grouped folds for both models",
        "folds": fold_summary,
        "training_cv": cv_results,
        "validation_random_forest": evaluate(
            validation[TARGET], probabilities
        ),
        "calibration": "No post-hoc calibration fitted",
        "threshold_policy": "Fixed 0.5; not optimized",
    }

    artifacts = ROOT / "artifacts"
    reports = ROOT / "reports"
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    joblib.dump(forest, artifacts / "random_forest.joblib")
    pd.DataFrame(
        {
            "row_id": validation_ids.to_numpy(),
            "churn": validation[TARGET].to_numpy(),
            "random_forest": probabilities,
        }
    ).to_csv(
        artifacts / "forest_validation_predictions.csv",
        index=False,
    )

    text = json.dumps(report, indent=2, allow_nan=False)
    (reports / "challenger_validation.json").write_text(
        text + "\n", encoding="utf-8"
    )
    print(text)


if __name__ == "__main__":
    main()