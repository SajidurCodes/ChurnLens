"""Inference using the exact frozen model and decision."""

import hashlib
import io
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnlens.features import MODEL_FEATURES
from churnlens.schemas import PredictionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Recorded by the completed final-test evaluation.
EXPECTED_RELEASE_SHA256 = (
    "67d7efb90eb15fa01b2f3d6f86ab3b0081a128fdd660fa4494397a714fe8acf6"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PredictionService:
    """Load once, then reuse the fitted pipeline for predictions."""

    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        root = Path(root).resolve()
        release_path = root / "reports" / "frozen_model_manifest.json"
        release_bytes = release_path.read_bytes()

        if digest(release_bytes) != EXPECTED_RELEASE_SHA256:
            raise ValueError("Frozen manifest checksum mismatch.")

        release = json.loads(release_bytes)

        if release["model_features"] != list(MODEL_FEATURES):
            raise ValueError("Frozen feature contract mismatch.")
        if release["comparison"] != ">=":
            raise ValueError("Unsupported decision comparison.")
        if platform.python_version() != release["python_version"]:
            raise ValueError("Python version differs from the frozen release.")

        for package, expected in release["package_versions"].items():
            if version(package) != expected:
                raise ValueError(f"Package version mismatch: {package}")

        verified_files = {}
        for name in ("model.joblib", "decision.json", "requirements.txt"):
            record = release["files"][name]
            path = (root / record["path"]).resolve()

            if not path.is_relative_to(root):
                raise ValueError("Frozen file path escapes the project.")

            content = path.read_bytes()
            if digest(content) != record["sha256"]:
                raise ValueError(f"Frozen file checksum mismatch: {name}")
            verified_files[name] = content

        decision = json.loads(verified_files["decision.json"])
        threshold = release["threshold"]

        if threshold != decision["selected_policy"]["threshold"]:
            raise ValueError("Threshold disagrees with the frozen decision.")
        if decision["selected_policy"]["comparison"] != ">=":
            raise ValueError("Decision comparison mismatch.")
        if decision["selected_model"] != release["selected_model"]:
            raise ValueError("Selected model mismatch.")
        if (
            decision["model_sha256"]
            != release["files"]["model.joblib"]["sha256"]
        ):
            raise ValueError("Model checksum disagrees with the decision.")

        if threshold is not None:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not np.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError("Invalid frozen threshold.")

        # Deserialize the exact verified bytes, not a second file read.
        # Only our own locally generated artifact is supported.
        model = joblib.load(io.BytesIO(verified_files["model.joblib"]))

        if list(model.feature_names_in_) != list(MODEL_FEATURES):
            raise ValueError("Model input feature order mismatch.")
        if list(model.classes_) != [0, 1]:
            raise ValueError("Unexpected model classes.")

        self._model = model
        self.model_version = release["release_id"]
        self.model_sha256 = release["files"]["model.joblib"]["sha256"]
        self.threshold = threshold

    def predict(self, payload: dict) -> dict:
        request = PredictionRequest.model_validate(payload)

        # Explicit column order matches training, regardless of JSON key order.
        frame = pd.DataFrame(
            [request.model_dump()],
            columns=list(MODEL_FEATURES),
        )

        probabilities = np.asarray(
            self._model.predict_proba(frame),
            dtype=float,
        )
        if (
            probabilities.shape != (1, 2)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise RuntimeError("Model returned invalid probabilities.")

        probability = float(probabilities[0, 1])
        flagged = (
            probability >= self.threshold
            if self.threshold is not None
            else False
        )

        return {
            "model_version": self.model_version,
            "churn_probability": probability,
            "threshold": self.threshold,
            "flagged_for_review": bool(flagged),
        }