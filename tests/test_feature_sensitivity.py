from copy import deepcopy

import pytest

from backend.src.survival_predictor import SurvivalPredictor


@pytest.fixture(scope="module")
def predictor():
    return SurvivalPredictor()


@pytest.fixture(scope="module")
def baseline_customer():
    return {
        "tenure": 12,
        "forecast_horizon": 6,
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }


def prediction_signature(result):
    curve_signature = tuple(
        (
            round(float(point["month"]), 4),
            round(float(point["retention"]), 6),
            round(float(point["churn"]), 6),
        )
        for point in result["survival_curve"]
    )

    return (
        round(
            float(
                result["hazard_ratio_multiplier"]
            ),
            8,
        ),
        round(
            float(
                result["projected_churn_pct"]
            ),
            8,
        ),
        round(
            float(
                result["projected_retention_pct"]
            ),
            8,
        ),
        curve_signature,
    )


def assert_prediction_changes(
    predictor,
    baseline_customer,
    feature,
    new_value,
):
    baseline = deepcopy(
        baseline_customer
    )

    changed = deepcopy(
        baseline_customer
    )

    changed[feature] = new_value

    baseline_result = (
        predictor.predict_risk_profile(
            baseline
        )
    )

    changed_result = (
        predictor.predict_risk_profile(
            changed
        )
    )

    baseline_signature = prediction_signature(
        baseline_result
    )

    changed_signature = prediction_signature(
        changed_result
    )

    assert baseline_signature != changed_signature, (
        f"Changing '{feature}' from "
        f"{baseline[feature]!r} to "
        f"{new_value!r} did not change "
        "the prediction."
    )


def test_tenure_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "tenure",
        48,
    )


def test_monthly_charges_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "MonthlyCharges",
        150.0,
    )


def test_total_charges_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "TotalCharges",
        3000.0,
    )


def test_contract_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "Contract",
        "Two year",
    )


def test_internet_service_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "InternetService",
        "DSL",
    )


def test_tech_support_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "TechSupport",
        "Yes",
    )


def test_online_security_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "OnlineSecurity",
        "Yes",
    )


def test_payment_method_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "PaymentMethod",
        "Credit card (automatic)",
    )


def test_paperless_billing_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "PaperlessBilling",
        "No",
    )


def test_gender_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "gender",
        "Female",
    )


def test_partner_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "Partner",
        "Yes",
    )


def test_dependents_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "Dependents",
        "Yes",
    )


def test_phone_service_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "PhoneService",
        "No",
    )


def test_multiple_lines_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "MultipleLines",
        "Yes",
    )


def test_online_backup_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "OnlineBackup",
        "Yes",
    )


def test_device_protection_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "DeviceProtection",
        "Yes",
    )


def test_streaming_tv_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "StreamingTV",
        "Yes",
    )


def test_streaming_movies_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "StreamingMovies",
        "Yes",
    )


def test_senior_citizen_changes_prediction(
    predictor,
    baseline_customer,
):
    assert_prediction_changes(
        predictor,
        baseline_customer,
        "SeniorCitizen",
        1,
    )