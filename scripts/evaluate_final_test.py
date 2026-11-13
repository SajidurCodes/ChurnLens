"""Evaluate the frozen model without fitting or selecting anything."""

import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from churnlens.features import MODEL_FEATURES, TARGET
from scripts.audit_data import (
    DATA_PATH,
    EXPECTED_SHA256,
    ROOT,
    normalize_column,
)
from scripts.compare_calibration import reliability_bins
from scripts.train_baselines import EXPECTED_MANIFEST_SHA256, evaluate


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_metrics(y, probabilities, threshold) -> dict:
    labels = np.asarray(y)
    p = np.asarray(probabilities, dtype=float)

    if labels.ndim != 1 or p.ndim != 1 or len(labels) != len(p):
        raise ValueError("Labels and probabilities must be matching vectors.")
    if len(p) == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be nonempty and binary.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities.")
    if threshold is not None:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid threshold.")

    flagged = (
        p >= threshold
        if threshold is not None
        else np.zeros(len(p), dtype=bool)
    )

    return {
        "threshold": threshold,
        "comparison": ">=",
        "flagged_rows": int(flagged.sum()),
        "flagged_fraction": float(flagged.mean()),
        "precision": float(precision_score(labels, flagged, zero_division=0)),
        "recall": float(recall_score(labels, flagged, zero_division=0)),
        "f1": float(f1_score(labels, flagged, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            labels, flagged, labels=[0, 1]
        ).tolist(),
    }


def main() -> None:
    report_path = ROOT / "reports" / "final_test.json"
    predictions_path = ROOT / "artifacts" / "final_test_predictions.csv"

    if report_path.exists() or predictions_path.exists():
        raise FileExistsError(
            "Final-test output already exists. Inspect it rather than "
            "silently replacing the original evaluation."
        )

    release_path = ROOT / "reports" / "frozen_model_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))

    if release["status"] != "frozen_before_final_test":
        raise ValueError("Unexpected release status.")
    if release["model_features"] != list(MODEL_FEATURES):
        raise ValueError("Feature contract differs from the frozen release.")
    if release["comparison"] != ">=":
        raise ValueError("Unsupported threshold comparison.")

    if platform.python_version() != release["python_version"]:
        raise ValueError("Python version differs from the frozen environment.")
    for package, expected in release["package_versions"].items():
        if version(package) != expected:
            raise ValueError(f"Package version changed: {package}")

    frozen_paths = {}
    for name, record in release["files"].items():
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen file checksum mismatch: {name}")
        frozen_paths[name] = path

    decision = json.loads(
        frozen_paths["decision.json"].read_text(encoding="utf-8")
    )
    threshold = release["threshold"]

    if threshold != decision["selected_policy"]["threshold"]:
        raise ValueError("Frozen threshold disagrees with the decision.")
    if release["selected_model"] != decision["selected_model"]:
        raise ValueError("Frozen model name disagrees with the decision.")
    if (
        release["files"]["model.joblib"]["sha256"]
        != decision["model_sha256"]
    ):
        raise ValueError("Frozen model disagrees with the decision.")
    if decision["selected_policy"]["comparison"] != ">=":
        raise ValueError("Decision uses a different comparison.")

    split_path = ROOT / "data" / "splits" / "manifest.csv"
    if not (
        sha256(DATA_PATH)
        == release["source_sha256"]
        == EXPECTED_SHA256
    ):
        raise ValueError("Source checksum mismatch.")
    if not (
        sha256(split_path)
        == release["split_manifest_sha256"]
        == EXPECTED_MANIFEST_SHA256
    ):
        raise ValueError("Split checksum mismatch.")

    frame = pd.read_csv(DATA_PATH).rename(columns=normalize_column)
    manifest = pd.read_csv(split_path)
    test_ids = manifest.loc[
        manifest["split"] == "test", "row_id"
    ].to_numpy()
    test = frame.iloc[test_ids].reset_index(drop=True)

    if len(test) != 631 or set(test[TARGET].unique()) != {0, 1}:
        raise ValueError("Unexpected final-test population.")

    # Only load the locally generated, checksum-verified artifact.
    model = joblib.load(frozen_paths["model.joblib"])
    positive_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        test[list(MODEL_FEATURES)]
    )[:, positive_column]

    selected_metrics = decision_metrics(
        test[TARGET], probabilities, threshold
    )
    reference_metrics = evaluate(test[TARGET], probabilities)

    report = {
        "release_id": release["release_id"],
        "evaluation_partition": "test",
        "frozen_manifest_sha256": sha256(release_path),
        "model_sha256": release["files"]["model.joblib"]["sha256"],
        "source_sha256": release["source_sha256"],
        "split_manifest_sha256": release["split_manifest_sha256"],
        "rows": len(test),
        "churners": int(test[TARGET].sum()),
        "churn_rate": float(test[TARGET].mean()),
        "probability_metrics": {
            name: reference_metrics[name]
            for name in (
                "average_precision",
                "roc_auc",
                "brier_score",
                "log_loss",
            )
        },
        "frozen_policy": selected_metrics,
        "reference_at_0_5": decision_metrics(
            test[TARGET], probabilities, 0.5
        ),
        "reliability_bins": reliability_bins(
            test[TARGET], probabilities
        ),
        "confusion_matrix_layout": "[[TN, FP], [FN, TP]]",
        "limitations": [
            "Single held-out dataset partition; no temporal validation.",
            "Matching input profiles were grouped, not verified identities.",
            "Threshold selected on validation; not adjusted on test.",
            "A fixed threshold does not enforce a future capacity limit.",
        ],
    }

    flagged = (
        probabilities >= threshold
        if threshold is not None
        else np.zeros(len(test), dtype=bool)
    )
    predictions = pd.DataFrame(
        {
            "row_id": test_ids,
            "churn": test[TARGET].to_numpy(),
            "probability": probabilities,
            "flagged": flagged,
        }
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(report, indent=2, allow_nan=False)
    predictions.to_csv(predictions_path, index=False, mode="x")
    with report_path.open("x", encoding="utf-8") as file:
        file.write(text + "\n")

    print(text)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()