from pathlib import Path

import numpy as np
import pytest
import time

from backend.app.prediction_service import PredictionService
from backend.src.survival_predictor import SurvivalPredictor
from backend.src.model_validator import ModelValidator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "models"
    / "cox_ph_model.pkl"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "data"
    / "processed"
    / "scaler.pkl"
)


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


def test_model_artifacts_are_valid():
    validator = ModelValidator(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
    )

    info = validator.validate_artifacts()

    assert info["valid"] is True
    assert info["model_features"] is not None
    assert info["scaler_features"] is not None
    assert (
        info["model_features"]
        == info["scaler_features"]
    )


def test_predictor_initializes(predictor):
    assert predictor is not None


def test_baseline_prediction_is_valid(
    predictor,
    baseline_customer,
):
    result = predictor.predict_risk_profile(
        baseline_customer
    )

    validator = ModelValidator(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
    )

    validator.validate_prediction_output(result)


def test_prediction_contains_required_business_outputs(
    predictor,
    baseline_customer,
):
    result = predictor.predict_risk_profile(
        baseline_customer
    )

    assert "hazard_ratio_multiplier" in result
    assert "projected_churn_pct" in result
    assert "projected_retention_pct" in result
    assert "risk_tier" in result
    assert "recommended_action" in result
    assert "survival_curve" in result


def test_hazard_is_positive(
    predictor,
    baseline_customer,
):
    result = predictor.predict_risk_profile(
        baseline_customer
    )

    hazard = float(
        result["hazard_ratio_multiplier"]
    )

    assert np.isfinite(hazard)
    assert hazard > 0


def test_churn_and_retention_are_valid(
    predictor,
    baseline_customer,
):
    result = predictor.predict_risk_profile(
        baseline_customer
    )

    churn = float(
        result["projected_churn_pct"]
    )

    retention = float(
        result["projected_retention_pct"]
    )

    assert 0 <= churn <= 100
    assert 0 <= retention <= 100

    assert np.isclose(
        churn + retention,
        100,
        atol=0.2,
    )


def test_survival_curve_is_valid(
    predictor,
    baseline_customer,
):
    result = predictor.predict_risk_profile(
        baseline_customer
    )

    curve = result["survival_curve"]

    assert isinstance(curve, list)
    assert len(curve) > 0

    months = [
        point["month"]
        for point in curve
    ]

    assert months == sorted(months)

    for point in curve:
        assert 0 <= point["retention"] <= 1
        assert 0 <= point["churn"] <= 1

        assert np.isclose(
            point["retention"]
            + point["churn"],
            1.0,
            atol=0.01,
        )


def test_forecast_target_is_correct(
    predictor,
    baseline_customer,
):
    result = predictor.predict_risk_profile(
        baseline_customer
    )

    expected_target = (
        baseline_customer["tenure"]
        + baseline_customer["forecast_horizon"]
    )

    assert (
        result["target_month"]
        == expected_target
    )


@pytest.fixture(scope="module")
def prediction_service():
    return PredictionService()


def test_prediction_contains_performance_in_development(
    prediction_service,
    baseline_customer,
):
    result = prediction_service.predict(baseline_customer)

    assert "performance" in result

    performance = result["performance"]

    assert "timings_ms" in performance
    assert "total_measured_ms" in performance

    assert performance["total_measured_ms"] >= 0
    assert isinstance(performance["timings_ms"], dict)
    assert "base_prediction" in performance["timings_ms"]
    assert "intervention_simulation" in performance["timings_ms"]
    assert "roi" in performance["timings_ms"]
    assert "retention_decision" in performance["timings_ms"]


def test_prediction_completes_within_reasonable_time(
    prediction_service,
    baseline_customer,
):
    start = time.perf_counter()

    result = prediction_service.predict(baseline_customer)

    elapsed = time.perf_counter() - start

    assert result["projected_churn"] >= 0
    assert result["projected_churn"] <= 100

    assert elapsed < 5.0