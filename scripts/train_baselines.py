"""Train initial baselines and evaluate on validation rows only."""

import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churnlens.features import MODEL_FEATURES, TARGET
from churnlens.models import build_logistic
from scripts.audit_data import (
    DATA_PATH,
    EXPECTED_SHA256,
    ROOT,
    normalize_column,
)


EXPECTED_MANIFEST_SHA256 = (
    "b7ad8e3c772fb1b2f1b095765cacdfc34b670eec80fa12b9539e2cc1f46e0151"
)


def evaluate(y: pd.Series, probabilities: np.ndarray) -> dict:
    predicted = (probabilities >= 0.5).astype(int)

    return {
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "threshold": 0.5,
        "precision": float(
            precision_score(y, predicted, zero_division=0)
        ),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "flagged_fraction": float(predicted.mean()),
        "confusion_matrix": confusion_matrix(
            y, predicted, labels=[0, 1]
        ).tolist(),
    }


def main() -> None:
    source_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if source_hash != EXPECTED_SHA256:
        raise ValueError("Source CSV checksum mismatch.")

    manifest_path = ROOT / "data" / "splits" / "manifest.csv"
    manifest_hash = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Split manifest checksum mismatch.")

    frame = pd.read_csv(DATA_PATH).rename(columns=normalize_column)
    manifest = pd.read_csv(manifest_path)

    def get_partition(name: str):
        row_ids = manifest.loc[manifest["split"] == name, "row_id"]
        part = frame.iloc[row_ids].copy()
        return part[list(MODEL_FEATURES)], part[TARGET]

    X_train, y_train = get_partition("train")
    X_validation, y_validation = get_partition("validation")

    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": build_logistic(),
    }

    results = {}
    validation_predictions = pd.DataFrame(
        {
            "row_id": manifest.loc[
                manifest["split"] == "validation", "row_id"
            ].to_numpy(),
            "churn": y_validation.to_numpy(),
        }
    )

    for name, estimator in models.items():
        # A non-converged logistic fit must not silently become a result.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(X_train, y_train)

        positive_column = list(estimator.classes_).index(1)
        probabilities = estimator.predict_proba(X_validation)[
            :, positive_column
        ]

        results[name] = evaluate(y_validation, probabilities)
        validation_predictions[name] = probabilities

    report = {
        "source_sha256": source_hash,
        "manifest_sha256": manifest_hash,
        "evaluation_partition": "validation",
        "train_rows": len(X_train),
        "validation_rows": len(X_validation),
        "training_churn_rate": float(y_train.mean()),
        "validation_churn_rate": float(y_validation.mean()),
        "calibration": "No post-hoc calibration fitted",
        "threshold_policy": "Fixed 0.5 baseline; not optimized",
        "confusion_matrix_layout": "[[TN, FP], [FN, TP]]",
        "models": results,
    }

    artifacts = ROOT / "artifacts"
    reports = ROOT / "reports"
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    for name, estimator in models.items():
        joblib.dump(estimator, artifacts / f"{name}.joblib")

    validation_predictions.to_csv(
        artifacts / "baseline_validation_predictions.csv",
        index=False,
    )

    text = json.dumps(report, indent=2, allow_nan=False)
    (reports / "baseline_validation.json").write_text(
        text + "\n", encoding="utf-8"
    )
    print(text)


if __name__ == "__main__":
    main()