"""Compare raw and sigmoid-calibrated models on validation data."""

import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from churnlens.calibration import build_calibrated, calibration_folds
from churnlens.features import MODEL_FEATURES, TARGET
from churnlens.models import build_forest, build_logistic
from scripts.audit_data import (
    DATA_PATH,
    EXPECTED_SHA256,
    ROOT,
    normalize_column,
)
from scripts.train_baselines import EXPECTED_MANIFEST_SHA256, evaluate


def reliability_bins(y, probabilities) -> list[dict]:
    """Five fixed-width bins; intervals are left-closed, except 1 is included."""
    labels = np.asarray(y, dtype=float)
    p = np.asarray(probabilities, dtype=float)

    if labels.ndim != 1 or p.ndim != 1 or len(labels) != len(p):
        raise ValueError("Labels and probabilities must be matching vectors.")
    if len(p) == 0:
        raise ValueError("Inputs cannot be empty.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary.")

    edges = np.linspace(0.0, 1.0, 6)
    # Boundary probabilities enter the bin on their right; 1 stays in bin 4.
    bin_ids = np.searchsorted(edges[1:-1], p, side="right")

    rows = []
    for index in range(5):
        mask = bin_ids == index
        count = int(mask.sum())
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": float(p[mask].mean()) if count else None,
                "observed_churn_rate": (
                    float(labels[mask].mean()) if count else None
                ),
            }
        )
    return rows


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

    def partition(name: str):
        ids = manifest.loc[manifest["split"] == name, "row_id"].to_numpy()
        rows = frame.iloc[ids].reset_index(drop=True)
        return ids, rows[list(MODEL_FEATURES)], rows[TARGET]

    _, X_train, y_train = partition("train")
    validation_ids, X_validation, y_validation = partition("validation")

    folds = calibration_folds(X_train, y_train)

    artifacts = ROOT / "artifacts" / "calibration"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    predictions = pd.DataFrame(
        {
            "row_id": validation_ids,
            "churn": y_validation.to_numpy(),
        }
    )
    results = {}

    factories = {
        "logistic": build_logistic,
        "forest": build_forest,
    }

    for family, factory in factories.items():
        for calibrated in (False, True):
            name = f"{family}_{'sigmoid' if calibrated else 'raw'}"
            print(f"Fitting {name}...", flush=True)

            model = (
                build_calibrated(factory(), folds)
                if calibrated
                else factory()
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(X_train, y_train)

            positive_column = list(model.classes_).index(1)
            p = model.predict_proba(X_validation)[:, positive_column]

            results[name] = {
                "metrics": evaluate(y_validation, p),
                "reliability_bins": reliability_bins(y_validation, p),
            }
            predictions[name] = p
            joblib.dump(model, artifacts / f"{name}.joblib")

    report = {
        "source_sha256": source_hash,
        "manifest_sha256": manifest_hash,
        "fit_partition": "train",
        "evaluation_partition": "validation",
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "calibration": {
            "method": "sigmoid",
            "ensemble": False,
            "folds": 3,
            "grouping_features": list(MODEL_FEATURES),
            "seed": 42,
        },
        "reliability_bin_policy": (
            "[lower, upper), except final bin includes 1.0; "
            "empty bins have null statistics"
        ),
        "models": results,
    }

    report_path = reports / "calibration_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    predictions.to_csv(
        artifacts / "validation_predictions.csv",
        index=False,
    )

    print("\nVALIDATION METRICS")
    table = pd.DataFrame(
        {name: result["metrics"] for name, result in results.items()}
    ).T
    print(
        table[
            ["average_precision", "roc_auc", "brier_score", "log_loss"]
        ].to_string()
    )

    print("\nRELIABILITY BINS")
    for name, result in results.items():
        print(f"\n{name}")
        print(
            pd.DataFrame(result["reliability_bins"]).to_string(index=False)
        )

    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()