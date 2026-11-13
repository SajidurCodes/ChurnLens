"""Shared feature definitions for splitting, training, and inference."""

TARGET = "churn"

NUMERIC_FEATURES = (
    "call_failure",
    "subscription_length",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
)

ORDINAL_FEATURES = (
    "charge_amount",
    "age_group",
)

CATEGORICAL_FEATURES = (
    "complains",
    "tariff_plan",
)

MODEL_FEATURES = (
    *NUMERIC_FEATURES,
    *ORDINAL_FEATURES,
    *CATEGORICAL_FEATURES,
)

EXCLUDED_FEATURES = (
    "status",
    "age",
    "customer_value",
)