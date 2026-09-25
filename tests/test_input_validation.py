import pytest

from backend.src.survival_predictor import SurvivalPredictor


@pytest.fixture(scope="module")
def predictor():
    return SurvivalPredictor()


@pytest.fixture
def valid_customer():
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


def test_zero_tenure_is_rejected(
    predictor,
    valid_customer,
):
    customer = dict(valid_customer)
    customer["tenure"] = 0

    with pytest.raises(
        (ValueError, Exception)
    ):
        predictor.predict_risk_profile(
            customer
        )


def test_negative_horizon_is_rejected(
    predictor,
    valid_customer,
):
    customer = dict(valid_customer)
    customer["forecast_horizon"] = -1

    with pytest.raises(
        (ValueError, Exception)
    ):
        predictor.predict_risk_profile(
            customer
        )


def test_negative_monthly_charges_are_rejected(
    predictor,
    valid_customer,
):
    customer = dict(valid_customer)
    customer["MonthlyCharges"] = -100

    with pytest.raises(
        (ValueError, Exception)
    ):
        predictor.predict_risk_profile(
            customer
        )


def test_missing_required_prediction_data_is_rejected(
    predictor,
    valid_customer,
):
    customer = dict(valid_customer)

    del customer["Contract"]

    with pytest.raises(Exception):
        predictor.predict_risk_profile(
            customer
        )