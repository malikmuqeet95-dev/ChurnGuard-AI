import io

import pandas as pd

from backend.src.batch_predictor import OUTPUT_COLUMNS, predict_batch
from backend.src.batch_prioritization import prioritize_batch_report
from backend.src.batch_schema import REQUIRED_BATCH_COLUMNS
from backend.src.batch_validator import validate_batch_dataframe


def make_customer(customer_id="C001"):
    return {
        "Customer ID": customer_id,
        "tenure": 12,
        "forecast_horizon": 6,
        "SeniorCitizen": 0,
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }


class FakePredictor:
    def predict_risk_profile(self, customer):
        return {
            "hazard_ratio_multiplier": 2.5,
            "projected_churn_pct": 60.0,
            "projected_retention_pct": 40.0,
            "risk_tier": "CRITICAL",
            "recommended_action": "Customer-success outreach",
            "target_month": 18,
        }


def test_required_batch_schema_is_complete():
    customer = make_customer()

    for column in REQUIRED_BATCH_COLUMNS:
        assert column in customer


def test_validator_accepts_valid_customer():
    dataframe = pd.DataFrame(
        [
            make_customer("C001"),
            make_customer("C002"),
        ]
    )

    result = validate_batch_dataframe(dataframe)

    assert result.valid
    assert len(result.dataframe) == 2


def test_batch_prediction_preserves_customer_count():
    dataframe = pd.DataFrame(
        [
            make_customer("C001"),
            make_customer("C002"),
            make_customer("C003"),
        ]
    )

    result = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    assert len(result) == 3


def test_batch_prediction_contains_expected_columns():
    dataframe = pd.DataFrame([make_customer()])

    result = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    for column in OUTPUT_COLUMNS:
        assert column in result.columns


def test_batch_prediction_contains_priority_columns():
    dataframe = pd.DataFrame([make_customer()])

    result = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    assert "Priority" in result.columns
    assert "Priority Score" in result.columns


def test_priority_score_is_bounded():
    dataframe = pd.DataFrame(
        [
            make_customer("C001"),
            make_customer("C002"),
        ]
    )

    result = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    assert result["Priority Score"].between(0, 100).all()


def test_customer_ids_remain_unique():
    dataframe = pd.DataFrame(
        [
            make_customer("C001"),
            make_customer("C002"),
            make_customer("C003"),
        ]
    )

    result = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    assert result["Customer ID"].is_unique


def test_prioritization_does_not_modify_original_dataframe():
    original = pd.DataFrame(
        [
            {
                "Customer ID": "C001",
                "Risk Tier": "CRITICAL",
                "Hazard Ratio": 2.5,
                "Churn Probability": 0.60,
                "Retention Probability": 0.40,
                "Revenue at Risk": 540.0,
                "Risk Window": "Within 18 months",
                "Recommended Action": "Customer-success outreach",
            }
        ]
    )

    original_columns = list(original.columns)

    result = prioritize_batch_report(original)

    assert list(original.columns) == original_columns
    assert "Priority" in result.columns


def test_multiple_customers_are_prioritized():
    dataframe = pd.DataFrame(
        [
            make_customer("C001"),
            make_customer("C002"),
            make_customer("C003"),
            make_customer("C004"),
            make_customer("C005"),
        ]
    )

    result = predict_batch(
        dataframe,
        predictor=FakePredictor(),
    )

    assert len(result) == 5
    assert result["Priority"].notna().all()