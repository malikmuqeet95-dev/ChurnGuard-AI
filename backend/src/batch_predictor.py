"""
Batch prediction engine for ChurnGuard AI v2.1.0.

This module reuses the existing SurvivalPredictor so that single-customer
and batch predictions share the same underlying ML logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .batch_prioritization import prioritize_batch_report
from .batch_validator import validate_batch_or_raise
from .survival_predictor import SurvivalPredictor


OUTPUT_COLUMNS = [
    "Customer ID",
    "Risk Tier",
    "Hazard Ratio",
    "Churn Probability",
    "Retention Probability",
    "Revenue at Risk",
    "Risk Window",
    "Recommended Action",
    "Priority",
    "Priority Score",
]


class BatchPredictionError(RuntimeError):
    """Raised when batch prediction fails."""


def _calculate_revenue_at_risk(
    monthly_charges: float,
    churn_probability: float,
) -> float:
    """
    Estimate revenue at risk over a 12-month planning horizon.

    This is a planning estimate, not a causal financial forecast.
    """

    annual_value = float(monthly_charges) * 12.0

    return round(
        annual_value * float(churn_probability),
        2,
    )


def _build_risk_window(
    target_month: int,
) -> str:
    """Create a human-readable forecast window."""

    return f"Within {target_month} months"


def predict_batch(
    dataframe: pd.DataFrame,
    predictor: SurvivalPredictor | None = None,
) -> pd.DataFrame:
    """
    Predict customer risk for an entire validated dataframe.

    Parameters
    ----------
    dataframe:
        Customer dataframe following the batch input contract.

    predictor:
        Optional existing SurvivalPredictor instance.

    Returns
    -------
    pandas.DataFrame
        Customer-level risk report.

    Raises
    ------
    BatchValidationError
        If the input dataframe fails validation.

    BatchPredictionError
        If a customer cannot be scored.
    """

    df = validate_batch_or_raise(dataframe)

    if predictor is None:
        predictor = SurvivalPredictor()

    results: list[dict[str, Any]] = []

    for row_index, row in df.iterrows():

        customer_id = str(row["Customer ID"])

        customer_input = {
            "tenure": int(row["tenure"]),
            "forecast_horizon": int(row["forecast_horizon"]),
            "SeniorCitizen": int(row["SeniorCitizen"]),
            "MonthlyCharges": float(row["MonthlyCharges"]),
            "TotalCharges": float(row["TotalCharges"]),
            "gender": str(row["gender"]),
            "Partner": str(row["Partner"]),
            "Dependents": str(row["Dependents"]),
            "PhoneService": str(row["PhoneService"]),
            "MultipleLines": str(row["MultipleLines"]),
            "InternetService": str(row["InternetService"]),
            "OnlineSecurity": str(row["OnlineSecurity"]),
            "OnlineBackup": str(row["OnlineBackup"]),
            "DeviceProtection": str(row["DeviceProtection"]),
            "TechSupport": str(row["TechSupport"]),
            "StreamingTV": str(row["StreamingTV"]),
            "StreamingMovies": str(row["StreamingMovies"]),
            "Contract": str(row["Contract"]),
            "PaperlessBilling": str(row["PaperlessBilling"]),
            "PaymentMethod": str(row["PaymentMethod"]),
        }

        try:
            prediction = predictor.predict_risk_profile(
                customer_input
            )
        except Exception as exc:
            raise BatchPredictionError(
                f"Prediction failed for customer "
                f"{customer_id} at row {row_index}: {exc}"
            ) from exc

        churn_probability = (
            float(prediction["projected_churn_pct"]) / 100.0
        )

        retention_probability = (
            float(prediction["projected_retention_pct"]) / 100.0
        )

        monthly_charges = float(
            row["MonthlyCharges"]
        )

        revenue_at_risk = _calculate_revenue_at_risk(
            monthly_charges=monthly_charges,
            churn_probability=churn_probability,
        )

        target_month = int(
            prediction["target_month"]
        )

        results.append(
            {
                "Customer ID": customer_id,
                "Risk Tier": prediction["risk_tier"],
                "Hazard Ratio": float(
                    prediction["hazard_ratio_multiplier"]
                ),
                "Churn Probability": round(
                    churn_probability,
                    4,
                ),
                "Retention Probability": round(
                    retention_probability,
                    4,
                ),
                "Revenue at Risk": revenue_at_risk,
                "Risk Window": _build_risk_window(
                    target_month
                ),
                "Recommended Action": prediction[
                    "recommended_action"
                ],
            }
        )

    report = pd.DataFrame(
        results,
        columns=OUTPUT_COLUMNS,
    )

    # Replace: return report
    return prioritize_batch_report(report)


def predict_batch_from_csv(
    input_path: str | Path,
    output_path: str | Path,
    predictor: SurvivalPredictor | None = None,
) -> pd.DataFrame:
    """
    Read a customer CSV, score the entire portfolio, and write the
    resulting customer risk report to CSV.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input customer CSV not found: {input_path}"
        )

    try:
        dataframe = pd.read_csv(input_path)
    except Exception as exc:
        raise BatchPredictionError(
            f"Unable to read customer CSV: {exc}"
        ) from exc

    report = predict_batch(
        dataframe=dataframe,
        predictor=predictor,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        output_path,
        index=False,
    )

    return report