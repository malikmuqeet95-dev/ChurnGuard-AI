"""
ChurnGuard AI — Batch Portfolio Prioritization

Ranks batch-scored customers for retention planning.

This module does NOT change the underlying survival model.
It only prioritizes already-generated batch predictions.
"""

from __future__ import annotations

import pandas as pd


PRIORITY_COLUMNS = [
    "Priority",
    "Priority Score",
]


def calculate_priority_score(row: pd.Series) -> float:
    """
    Calculate a planning priority score from existing model outputs.

    Components:
    - churn probability: strongest signal
    - revenue at risk: business exposure
    - risk tier: additional risk weighting

    This is a decision-support score, not a causal or financial forecast.
    """

    churn_probability = float(row["Churn Probability"])
    revenue_at_risk = float(row["Revenue at Risk"])

    risk_tier = str(row["Risk Tier"]).upper()

    tier_weight = {
        "CRITICAL": 1.00,
        "MODERATE": 0.65,
        "HEALTHY": 0.25,
    }.get(risk_tier, 0.25)

    # Normalize revenue contribution with a simple logarithmic scale.
    revenue_component = min(
        1.0,
        max(0.0, revenue_at_risk / 5000.0),
    )

    score = (
        (churn_probability * 0.60)
        + (revenue_component * 0.25)
        + (tier_weight * 0.15)
    )

    return round(score * 100, 2)


def assign_priority(score: float) -> str:
    """
    Convert the numerical planning score into a priority bucket.
    """

    if score >= 70:
        return "URGENT"

    if score >= 45:
        return "HIGH"

    if score >= 20:
        return "MEDIUM"

    return "LOW"


def prioritize_batch_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add priority information and sort the portfolio.

    Highest-priority customers appear first.

    Existing columns are preserved.
    """

    if dataframe.empty:
        result = dataframe.copy()
        result["Priority"] = pd.Series(dtype=str)
        result["Priority Score"] = pd.Series(dtype=float)
        return result

    result = dataframe.copy()

    result["Priority Score"] = result.apply(
        calculate_priority_score,
        axis=1,
    )

    result["Priority"] = result["Priority Score"].apply(
        assign_priority,
    )

    priority_order = {
        "URGENT": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }

    result["_priority_order"] = result["Priority"].map(
        priority_order
    )

    result = result.sort_values(
        by=[
            "_priority_order",
            "Priority Score",
            "Revenue at Risk",
        ],
        ascending=[True, False, False],
    )

    result = result.drop(
        columns=["_priority_order"]
    )

    result = result.reset_index(drop=True)

    return result