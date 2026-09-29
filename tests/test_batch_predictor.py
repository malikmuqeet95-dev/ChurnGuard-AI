import pandas as pd

from backend.src.batch_predictor import (
    OUTPUT_COLUMNS,
    predict_batch,
    predict_batch_from_csv,
)
from tests.test_batch_validator import make_valid_customer


class FakePredictor:
    """Small deterministic predictor for batch unit tests."""

    def predict_risk_profile(self, customer_input):
        return {
            "target_month": (
                customer_input["tenure"]
                + customer_input["forecast_horizon"]
            ),
            "hazard_ratio_multiplier": 2.5,
            "projected_churn_pct": 70.0,
            "projected_retention_pct": 30.0,
            "risk_tier": "CRITICAL",
            "recommended_action": (
                "Retention incentive / customer-success outreach"
            ),
            "survival_curve": [],
        }


def test_batch_prediction_generates_expected_columns():
    dataframe = pd.DataFrame([
        make_valid_customer("CUST-001"),
        make_valid_customer("CUST-002"),
    ])

    report = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    assert len(report) == 2

    assert list(report.columns) == OUTPUT_COLUMNS

    assert report["Customer ID"].tolist() == [
        "CUST-001",
        "CUST-002",
    ]

    assert report["Risk Tier"].tolist() == [
        "CRITICAL",
        "CRITICAL",
    ]


def test_batch_prediction_calculates_probabilities():
    dataframe = pd.DataFrame([
        make_valid_customer("CUST-001"),
    ])

    report = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    row = report.iloc[0]

    assert row["Hazard Ratio"] == 2.5
    assert row["Churn Probability"] == 0.70
    assert row["Retention Probability"] == 0.30


def test_batch_prediction_calculates_revenue_at_risk():
    customer = make_valid_customer("CUST-001")
    customer["MonthlyCharges"] = 100.0

    dataframe = pd.DataFrame([customer])

    report = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    # 100 * 12 * 0.70 = 840
    assert report.iloc[0]["Revenue at Risk"] == 840.0


def test_batch_prediction_builds_risk_window():
    dataframe = pd.DataFrame([
        make_valid_customer("CUST-001"),
    ])

    report = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    assert report.iloc[0]["Risk Window"] == "Within 9 months"


def test_batch_prediction_from_csv_writes_report(tmp_path):
    input_path = tmp_path / "customers.csv"
    output_path = tmp_path / "customer_risk_report.csv"

    dataframe = pd.DataFrame([
        make_valid_customer("CUST-001"),
        make_valid_customer("CUST-002"),
    ])

    dataframe.to_csv(
        input_path,
        index=False,
    )

    report = predict_batch_from_csv(
        input_path=input_path,
        output_path=output_path,
        predictor=FakePredictor(),
    )

    assert output_path.exists()

    saved_report = pd.read_csv(output_path)

    assert len(saved_report) == 2
    assert list(saved_report.columns) == OUTPUT_COLUMNS

    assert report.equals(saved_report)


def test_batch_prediction_rejects_invalid_input():
    customer = make_valid_customer("CUST-001")
    customer["Contract"] = "INVALID"

    dataframe = pd.DataFrame([customer])

    try:
        predict_batch(
            dataframe,
            predictor=FakePredictor(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid batch input should have been rejected."
        )