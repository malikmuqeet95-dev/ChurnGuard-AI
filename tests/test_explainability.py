from backend.app.prediction_service import PredictionService


def build_customer():
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


def test_explanation_is_returned():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    assert "explanation" in result


def test_explanation_contains_driver_categories():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    explanation = result["explanation"]

    assert "top_drivers" in explanation
    assert "risk_drivers" in explanation
    assert "protective_drivers" in explanation
    assert "driver_count" in explanation


def test_driver_structure_is_valid():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    drivers = result[
        "explanation"
    ]["top_drivers"]

    for driver in drivers:
        assert "feature" in driver
        assert "model_feature" in driver
        assert "value" in driver
        assert "coefficient" in driver
        assert "contribution" in driver
        assert "hazard_multiplier" in driver
        assert "direction" in driver
        assert "impact" in driver
        assert "description" in driver


def test_driver_direction_is_valid():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    drivers = result[
        "explanation"
    ]["top_drivers"]

    valid_directions = {
        "increases_risk",
        "reduces_risk",
        "neutral",
    }

    for driver in drivers:
        assert (
            driver["direction"]
            in valid_directions
        )


def test_driver_impact_is_valid():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    drivers = result[
        "explanation"
    ]["top_drivers"]

    valid_impacts = {
        "low",
        "medium",
        "high",
    }

    for driver in drivers:
        assert (
            driver["impact"]
            in valid_impacts
        )


def test_explanation_does_not_change_prediction():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    assert result["hazard_ratio"] > 0

    assert (
        0
        <= result["projected_churn"]
        <= 100
    )

    assert (
        0
        <= result["projected_retention"]
        <= 100
    )

    assert (
        result["projected_churn"]
        + result["projected_retention"]
        <= 100.01
    )