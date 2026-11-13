"""Freeze the selected model and decision before final-test evaluation."""

import hashlib
import json
import platform
import shutil
from importlib.metadata import version

from churnlens.features import MODEL_FEATURES
from scripts.audit_data import (
    DATA_PATH,
    EXPECTED_SHA256,
    ROOT,
)
from scripts.train_baselines import EXPECTED_MANIFEST_SHA256


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    decision_path = ROOT / "reports" / "threshold_validation.json"
    analysis_path = ROOT / "reports" / "validation_error_analysis.json"
    split_path = ROOT / "data" / "splits" / "manifest.csv"
    requirements_path = ROOT / "requirements.txt"

    frozen_dir = ROOT / "artifacts" / "frozen"
    manifest_path = ROOT / "reports" / "frozen_model_manifest.json"

    if frozen_dir.exists() or manifest_path.exists():
        raise FileExistsError(
            "Frozen output already exists. Do not replace it silently."
        )

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    source_model = ROOT / decision["model_path"]

    if sha256(DATA_PATH) != EXPECTED_SHA256:
        raise ValueError("Source CSV changed.")
    if sha256(split_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Split assignments changed.")
    if sha256(source_model) != decision["model_sha256"]:
        raise ValueError("Selected model changed.")
    if sha256(decision_path) != analysis["decision_sha256"]:
        raise ValueError("Decision changed after error analysis.")
    if analysis["model_sha256"] != decision["model_sha256"]:
        raise ValueError("Error analysis used a different model.")
    if analysis["threshold"] != decision["selected_policy"]["threshold"]:
        raise ValueError("Error analysis used a different threshold.")
    if decision["selected_model"] != "forest_raw":
        raise ValueError("Unexpected selected model.")
    if decision["selection_partition"] != "validation":
        raise ValueError("Decision must come from validation.")
    if not requirements_path.is_file():
        raise FileNotFoundError("requirements.txt is missing.")

    sources = {
        "model.joblib": source_model,
        "decision.json": decision_path,
        "requirements.txt": requirements_path,
    }

    frozen_dir.mkdir(parents=True, exist_ok=False)

    file_records = {}
    for filename, source in sources.items():
        destination = frozen_dir / filename
        shutil.copy2(source, destination)

        if sha256(destination) != sha256(source):
            raise ValueError(f"Copy verification failed: {filename}")

        file_records[filename] = {
            "path": destination.relative_to(ROOT).as_posix(),
            "sha256": sha256(destination),
        }

    manifest = {
        "release_id": "churnlens-v0.1",
        "status": "frozen_before_final_test",
        "selected_model": decision["selected_model"],
        "threshold": decision["selected_policy"]["threshold"],
        "comparison": decision["selected_policy"]["comparison"],
        "model_features": list(MODEL_FEATURES),
        "source_sha256": EXPECTED_SHA256,
        "split_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "validation_error_analysis_sha256": sha256(analysis_path),
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package)
            for package in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
            )
        },
        "files": file_records,
        "limitations": [
            "Offline dataset benchmark; no verified temporal test split.",
            "Profile grouping does not establish unique customer identities.",
            "Fixed threshold does not guarantee a future 20% flag rate.",
            "Final-test performance has not yet been evaluated.",
        ],
    }

    text = json.dumps(manifest, indent=2, allow_nan=False)
    with manifest_path.open("x", encoding="utf-8") as file:
        file.write(text + "\n")

    print(text)
    print(f"\nFrozen bundle: {frozen_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()