"""Model factories: each call returns an unfitted estimator."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnlens.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
)


def build_preprocessing(scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler() if scale_numeric else "passthrough",
                list(NUMERIC_FEATURES + ORDINAL_FEATURES),
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="error",
                    sparse_output=False,
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )


def build_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=None,
                ),
            ),
        ]
    )


def build_forest() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight=None,
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )