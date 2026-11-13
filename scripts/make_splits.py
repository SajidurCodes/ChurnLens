"""Create the fixed split manifest for the verified source snapshot."""

import hashlib
import json

import pandas as pd
import sklearn

from churnlens.features import MODEL_FEATURES, TARGET
from churnlens.splitting import SPLIT_SEED, make_manifest
from scripts.audit_data import (
    DATA_PATH,
    EXPECTED_SHA256,
    ROOT,
    audit_frame,
    normalize_column,
)


MANIFEST_PATH = ROOT / "data" / "splits" / "manifest.csv"
REPORT_PATH = ROOT / "reports" / "split_summary.json"


def main() -> None:
    if MANIFEST_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError(
            "Split output already exists. Inspect it before replacing anything."
        )

    checksum = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise ValueError("Source checksum mismatch.")

    raw = pd.read_csv(DATA_PATH)
    audit_frame(raw)

    frame = raw.rename(columns=normalize_column)
    manifest = make_manifest(frame)

    summary = {
        "source_sha256": checksum,
        "seed": SPLIT_SEED,
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "method": "StratifiedGroupKFold: test=0, validation=1, train=2/3/4",
        "grouping_features": list(MODEL_FEATURES),
        "unique_input_profiles": int(manifest["group_id"].nunique()),
        "profiles_crossing_splits": int(
            (manifest.groupby("group_id")["split"].nunique() > 1).sum()
        ),
        "splits": {},
    }

    for name in ("train", "validation", "test"):
        part = manifest.loc[manifest["split"] == name]
        labels = frame.iloc[part["row_id"]][TARGET]

        summary["splits"][name] = {
            "rows": len(part),
            "fraction": len(part) / len(frame),
            "groups": int(part["group_id"].nunique()),
            "non_churn": int((labels == 0).sum()),
            "churn": int((labels == 1).sum()),
            "churn_rate": float(labels.mean()),
        }

    manifest_text = manifest.to_csv(index=False, lineterminator="\n")
    summary["manifest_sha256"] = hashlib.sha256(
        manifest_text.encode("utf-8")
    ).hexdigest()

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("x", encoding="utf-8", newline="") as file:
        file.write(manifest_text)

    report_text = json.dumps(summary, indent=2, allow_nan=False)
    with REPORT_PATH.open("x", encoding="utf-8") as file:
        file.write(report_text + "\n")

    print(report_text)
    print(f"\nSaved manifest: {MANIFEST_PATH}")
    print(f"Saved summary: {REPORT_PATH}")


if __name__ == "__main__":
    main()