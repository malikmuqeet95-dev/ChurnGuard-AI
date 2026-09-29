import pandas as pd

from backend.src.batch_prioritization import (
    assign_priority,
    calculate_priority_score,
    prioritize_batch_report,
)


def make_report():
    return pd.DataFrame(
        [
            {
                "Customer ID": "C001",
                "Risk Tier": "CRITICAL",
                "Hazard Ratio": 3.0,
                "Churn Probability": 0.90,
                "Retention Probability": 0.10,
                "Revenue at Risk": 810.0,
                "Risk Window": "Within 9 months",
                "Recommended Action": "Customer-success outreach",
            },
            {
                "Customer ID": "C002",
                "Risk Tier": "MODERATE",
                "Hazard Ratio": 1.5,
                "Churn Probability": 0.50,
                "Retention Probability": 0.50,
                "Revenue at Risk": 300.0,
                "Risk Window": "Within 9 months",
                "Recommended Action": "Feature engagement onboarding",
            },
            {
                "Customer ID": "C003",
                "Risk Tier": "HEALTHY",
                "Hazard Ratio": 0.8,
                "Churn Probability": 0.10,
                "Retention Probability": 0.90,
                "Revenue at Risk": 60.0,
                "Risk Window": "Within 9 months",
                "Recommended Action": "Loyalty rewards",
            },
        ]
    )


def test_priority_score_is_numeric():
    report = make_report()

    score = calculate_priority_score(report.iloc[0])

    assert isinstance(score, float)
    assert 0 <= score <= 100


def test_priority_buckets():
    assert assign_priority(80) == "URGENT"
    assert assign_priority(60) == "HIGH"
    assert assign_priority(30) == "MEDIUM"
    assert assign_priority(10) == "LOW"


def test_prioritization_adds_columns():
    result = prioritize_batch_report(make_report())

    assert "Priority" in result.columns
    assert "Priority Score" in result.columns


def test_prioritization_sorts_highest_first():
    result = prioritize_batch_report(make_report())

    assert result.iloc[0]["Customer ID"] == "C001"


def test_existing_prediction_columns_are_preserved():
    original = make_report()
    result = prioritize_batch_report(original)

    for column in original.columns:
        assert column in result.columns


def test_empty_dataframe():
    empty = make_report().iloc[0:0]

    result = prioritize_batch_report(empty)

    assert result.empty
    assert "Priority" in result.columns
    assert "Priority Score" in result.columns