"""Audit the source CSV without modifying it."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "customer_churn.csv"
REPORT_PATH = ROOT / "reports" / "data_audit.json"

EXPECTED_SHA256 = (
    "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1"
)

EXPECTED_COLUMNS = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "status",
    "age",
    "customer_value",
    "churn",
]


def normalize_column(name: str) -> str:
    """Collapse whitespace and use consistent lowercase names."""
    return "_".join(name.strip().lower().split())


def audit_frame(raw: pd.DataFrame) -> dict:
    """Validate a frame and return JSON-compatible audit results."""
    df = raw.copy()
    df.columns = [normalize_column(str(name)) for name in df.columns]

    if df.columns.duplicated().any():
        raise ValueError("Column names collide after normalization.")

    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    unexpected = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            f"Schema mismatch: missing={missing}, unexpected={unexpected}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_cells = int(df.isna().sum().sum())
    if missing_cells:
        raise ValueError(f"Dataset contains {missing_cells} missing cells.")

    for column in EXPECTED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite or non-finite values.")

    if (df < 0).any().any():
        raise ValueError("Dataset contains negative values.")

    integer_columns = [
        column for column in EXPECTED_COLUMNS
        if column != "customer_value"
    ]
    if (df[integer_columns] % 1 != 0).any().any():
        raise ValueError("An integer-valued field contains a fraction.")

    allowed_values = {
        "churn": {0, 1},
        "complains": {0, 1},
        "tariff_plan": {1, 2},
        "status": {1, 2},
        "age_group": {1, 2, 3, 4, 5},
    }
    for column, allowed in allowed_values.items():
        observed = set(df[column].unique())
        if not observed.issubset(allowed):
            raise ValueError(f"Invalid category in {column}: {observed}")

    if set(df["churn"].unique()) != {0, 1}:
        raise ValueError("The source dataset must contain both target classes.")

    feature_columns = [
        column for column in EXPECTED_COLUMNS if column != "churn"
    ]
    label_counts = df.groupby(
        feature_columns, dropna=False
    )["churn"].nunique()

    warnings = []
    if df["charge_amount"].max() > 9:
        warnings.append(
            "Charge Amount exceeds the documented 0–9 range. "
            "Preserve the source values; investigate and document."
        )

    return {
        "rows": len(df),
        "input_columns": len(feature_columns),
        "normalized_columns": df.columns.tolist(),
        "missing_cells": missing_cells,
        "class_counts": {
            str(label): int((df["churn"] == label).sum())
            for label in (0, 1)
        },
        "churn_rate": float(df["churn"].mean()),
        "duplicate_rows_beyond_first": int(df.duplicated().sum()),
        "duplicate_input_profiles_beyond_first": int(
            df.duplicated(subset=feature_columns).sum()
        ),
        "conflicting_label_groups": int((label_counts > 1).sum()),
        "ranges": {
            column: {
                "min": float(df[column].min()),
                "max": float(df[column].max()),
            }
            for column in EXPECTED_COLUMNS
        },
        "warnings": warnings,
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    checksum = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise ValueError("Dataset checksum does not match our source snapshot.")

    report = audit_frame(pd.read_csv(DATA_PATH))
    report["sha256"] = checksum

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, indent=2, allow_nan=False)
    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")

    print(report_text)
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()