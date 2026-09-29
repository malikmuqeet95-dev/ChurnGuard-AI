import io

import pandas as pd

from backend.src.batch_predictor import predict_batch


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


def make_customer(customer_id):
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


def test_v21_final_report_contract():
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

    required_columns = [
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

    assert list(result.columns) == required_columns


def test_v21_report_has_no_missing_values():
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

    assert not result.isnull().any().any()


def test_v21_report_has_valid_risk_values():
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

    assert result["Risk Tier"].isin(
        ["CRITICAL", "MODERATE", "HEALTHY"]
    ).all()

    assert result["Churn Probability"].between(
        0, 1
    ).all()

    assert result["Retention Probability"].between(
        0, 1
    ).all()

    assert (result["Revenue at Risk"] >= 0).all()

    assert result["Priority"].isin(
        ["URGENT", "HIGH", "MEDIUM", "LOW"]
    ).all()

    assert result["Priority Score"].between(
        0, 100
    ).all()


def test_v21_customer_ids_are_unique():
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


def test_v21_csv_round_trip():
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

    buffer = io.StringIO()
    result.to_csv(buffer, index=False)

    buffer.seek(0)

    reloaded = pd.read_csv(buffer)

    assert list(reloaded.columns) == list(result.columns)
    assert len(reloaded) == len(result)
    assert reloaded["Customer ID"].tolist() == [
        "C001",
        "C002",
    ]