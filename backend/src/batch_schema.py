"""
Batch prediction input contract for ChurnGuard AI v2.1.0.

This module defines the required CSV structure for portfolio-level
customer risk prediction.
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Customer identification
# ---------------------------------------------------------------------------

CUSTOMER_ID_COLUMN: Final[str] = "Customer ID"


# ---------------------------------------------------------------------------
# Prediction configuration
# ---------------------------------------------------------------------------

PREDICTION_COLUMNS: Final[list[str]] = [
    "tenure",
    "forecast_horizon",
]


# ---------------------------------------------------------------------------
# Numeric customer features
# ---------------------------------------------------------------------------

NUMERIC_FEATURE_COLUMNS: Final[list[str]] = [
    "SeniorCitizen",
    "MonthlyCharges",
    "TotalCharges",
]


# ---------------------------------------------------------------------------
# Categorical customer features
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURE_COLUMNS: Final[list[str]] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


# ---------------------------------------------------------------------------
# Complete required input schema
# ---------------------------------------------------------------------------

REQUIRED_BATCH_COLUMNS: Final[list[str]] = [
    CUSTOMER_ID_COLUMN,
    *PREDICTION_COLUMNS,
    *NUMERIC_FEATURE_COLUMNS,
    *CATEGORICAL_FEATURE_COLUMNS,
]


# ---------------------------------------------------------------------------
# Allowed categorical values
# ---------------------------------------------------------------------------

ALLOWED_CATEGORIES: Final[dict[str, set[str]]] = {
    "gender": {
        "Male",
        "Female",
    },
    "Partner": {
        "Yes",
        "No",
    },
    "Dependents": {
        "Yes",
        "No",
    },
    "PhoneService": {
        "Yes",
        "No",
    },
    "MultipleLines": {
        "Yes",
        "No",
        "No phone service",
    },
    "InternetService": {
        "DSL",
        "Fiber optic",
        "No",
    },
    "OnlineSecurity": {
        "Yes",
        "No",
        "No internet service",
    },
    "OnlineBackup": {
        "Yes",
        "No",
        "No internet service",
    },
    "DeviceProtection": {
        "Yes",
        "No",
        "No internet service",
    },
    "TechSupport": {
        "Yes",
        "No",
        "No internet service",
    },
    "StreamingTV": {
        "Yes",
        "No",
        "No internet service",
    },
    "StreamingMovies": {
        "Yes",
        "No",
        "No internet service",
    },
    "Contract": {
        "Month-to-month",
        "One year",
        "Two year",
    },
    "PaperlessBilling": {
        "Yes",
        "No",
    },
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


# ---------------------------------------------------------------------------
# Validation ranges
# ---------------------------------------------------------------------------

TENURE_MIN: Final[int] = 1
TENURE_MAX: Final[int] = 72

FORECAST_HORIZON_MIN: Final[int] = 1
FORECAST_HORIZON_MAX: Final[int] = 36

MONTHLY_CHARGES_MIN: Final[float] = 10.0
MONTHLY_CHARGES_MAX: Final[float] = 250.0

TOTAL_CHARGES_MIN: Final[float] = 0.0

SENIOR_CITIZEN_VALUES: Final[set[int]] = {
    0,
    1,
}