"""Describe validation errors using the selected model and threshold."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from churnlens.features import MODEL_FEATURES, TARGET
from scripts.audit_data import (
    DATA_PATH,
    EXPECTED_SHA256,
    ROOT,
    normalize_column,
)
from scripts.train_baselines import EXPECTED_MANIFEST_SHA256


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_slice(name: str, value: str, part: pd.DataFrame) -> dict:
    positive = part[TARGET] == 1
    flagged = part["flagged"]

    tp = int((positive & flagged).sum())
    fp = int((~positive & flagged).sum())
    fn = int((positive & ~flagged).sum())
    tn = int((~positive & ~flagged).sum())

    return {
        "slice": name,
        "value": value,
        "rows": len(part),
        "churners": tp + fn,
        "non_churners": tn + fp,
        "flagged": tp + fp,
        "false_negatives": fn,
        "false_positives": fp,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "small_sample": len(part) < 30 or tp + fn < 10 or tn + fp < 10,
    }


def main() -> None:
    decision_path = ROOT / "reports" / "threshold_validation.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    manifest_path = ROOT / "data" / "splits" / "manifest.csv"
    model_path = ROOT / decision["model_path"]

    if sha256(DATA_PATH) != EXPECTED_SHA256:
        raise ValueError("Source checksum mismatch.")
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Manifest checksum mismatch.")
    if sha256(model_path) != decision["model_sha256"]:
        raise ValueError("Selected model artifact has changed.")
    if decision["selection_partition"] != "validation":
        raise ValueError("Expected a validation-only decision.")
    if decision["selected_policy"]["comparison"] != ">=":
        raise ValueError("Unsupported threshold comparison.")

    frame = pd.read_csv(DATA_PATH).rename(columns=normalize_column)
    manifest = pd.read_csv(manifest_path)
    ids = manifest.loc[manifest["split"] == "validation", "row_id"].to_numpy()

    validation = frame.iloc[ids].copy().reset_index(drop=True)
    validation.insert(0, "row_id", ids)

    model = joblib.load(model_path)
    positive_column = list(model.classes_).index(1)
    validation["probability"] = model.predict_proba(
        validation[list(MODEL_FEATURES)]
    )[:, positive_column]

    threshold = decision["selected_policy"]["threshold"]
    validation["flagged"] = (
        validation["probability"] >= threshold
        if threshold is not None
        else False
    )

    actual_matrix = confusion_matrix(
        validation[TARGET],
        validation["flagged"].astype(int),
        labels=[0, 1],
    ).tolist()

    if actual_matrix != decision["selected_policy"]["confusion_matrix"]:
        raise ValueError("Predictions do not reproduce the selected decision.")

    validation["outcome"] = np.select(
        [
            (validation[TARGET] == 1) & ~validation["flagged"],
            (validation[TARGET] == 0) & validation["flagged"],
            (validation[TARGET] == 1) & validation["flagged"],
        ],
        ["false_negative", "false_positive", "true_positive"],
        default="true_negative",
    )

    validation["tenure_band"] = pd.cut(
        validation["subscription_length"],
        bins=[-np.inf, 12, 24, np.inf],
        labels=["up_to_12_months", "13_to_24_months", "over_24_months"],
    )
    validation["usage_band"] = pd.cut(
        validation["seconds_of_use"],
        bins=[-np.inf, 0, 1000, np.inf],
        labels=["zero_seconds", "1_to_1000_seconds", "over_1000_seconds"],
    )

    slices = []
    for column in ("tariff_plan", "age_group", "tenure_band", "usage_band"):
        for value, part in validation.groupby(column, observed=True):
            slices.append(summarize_slice(column, str(value), part))

    error_columns = [
        "row_id",
        *MODEL_FEATURES,
        TARGET,
        "probability",
        "flagged",
        "outcome",
    ]
    errors = validation.loc[
        validation["outcome"].isin(["false_negative", "false_positive"]),
        error_columns,
    ].copy()

    # Put the most confidently wrong predictions first.
    errors["wrong_class_confidence"] = np.where(
        errors[TARGET] == 1,
        1.0 - errors["probability"],
        errors["probability"],
    )
    errors = errors.sort_values(
        ["wrong_class_confidence", "row_id"],
        ascending=[False, True],
    )

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    errors.to_csv(artifacts / "validation_errors.csv", index=False)

    report = {
        "partition": "validation",
        "decision_sha256": sha256(decision_path),
        "model_sha256": sha256(model_path),
        "threshold": threshold,
        "confusion_matrix": actual_matrix,
        "error_rows": len(errors),
        "small_sample_rule": (
            "rows < 30 OR churners < 10 OR non-churners < 10; "
            "a descriptive caution, not a statistical test"
        ),
        "slices": slices,
    }
    report_path = ROOT / "reports" / "validation_error_analysis.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Reproduced selected confusion matrix:", actual_matrix)
    print("Total errors:", len(errors))
    print("\nVALIDATION SLICES")
    print(pd.DataFrame(slices).round(4).to_string(index=False))

    print("\nFIVE MOST CONFIDENTLY WRONG PREDICTIONS")
    print(
        errors[
            ["row_id", TARGET, "probability", "outcome"]
        ].head(5).to_string(index=False)
    )
    print(f"\nSaved report: {report_path}")
    print(f"Saved error rows: {artifacts / 'validation_errors.csv'}")


if __name__ == "__main__":
    main()