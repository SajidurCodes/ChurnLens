"""Select the raw forest's threshold using validation data only."""

import hashlib
import json

import joblib
import pandas as pd

from churnlens.features import MODEL_FEATURES, TARGET
from churnlens.thresholds import select_threshold
from scripts.audit_data import (
    DATA_PATH,
    EXPECTED_SHA256,
    ROOT,
    normalize_column,
)
from scripts.train_baselines import EXPECTED_MANIFEST_SHA256, evaluate


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = ROOT / "data" / "splits" / "manifest.csv"
    model_path = ROOT / "artifacts" / "calibration" / "forest_raw.joblib"
    report_path = ROOT / "reports" / "threshold_validation.json"

    if report_path.exists():
        raise FileExistsError(
            "Threshold report already exists. Inspect it before replacing it."
        )

    if file_sha256(DATA_PATH) != EXPECTED_SHA256:
        raise ValueError("Source checksum mismatch.")
    if file_sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Manifest checksum mismatch.")

    frame = pd.read_csv(DATA_PATH).rename(columns=normalize_column)
    manifest = pd.read_csv(manifest_path)
    row_ids = manifest.loc[
        manifest["split"] == "validation", "row_id"
    ].to_numpy()
    validation = frame.iloc[row_ids]

    # Load only our own locally generated model artifact.
    model = joblib.load(model_path)
    positive_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        validation[list(MODEL_FEATURES)]
    )[:, positive_column]
    labels = validation[TARGET]

    decision = select_threshold(
        labels,
        probabilities,
        max_flag_fraction=0.20,
    )

    report = {
        "selected_model": "forest_raw",
        "model_path": model_path.relative_to(ROOT).as_posix(),
        "model_sha256": file_sha256(model_path),
        "source_sha256": EXPECTED_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "selection_partition": "validation",
        "validation_rows": len(validation),
        "model_selection_basis": (
            "Lowest validation log loss among the four fixed candidates; "
            "also lowest validation Brier score."
        ),
        "threshold_objective": (
            "Maximize true positives within 20% validation capacity; "
            "break equal-recall ties by fewer flags; keep score ties together."
        ),
        "reference_at_0_5": evaluate(labels, probabilities),
        "selected_policy": decision,
        "confusion_matrix_layout": "[[TN, FP], [FN, TP]]",
        "limitation": (
            "The fixed threshold does not guarantee a 20% flag rate "
            "on another dataset."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False)
    with report_path.open("x", encoding="utf-8") as file:
        file.write(text + "\n")

    print(text)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()