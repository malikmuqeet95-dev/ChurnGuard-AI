import pytest

from backend.app.prediction_service import (
    PredictionService,
)


@pytest.fixture(scope="module")
def service():
    return PredictionService()


@pytest.fixture
def customer():
    return {
        "tenure": 3,
        "forecast_horizon": 6,
        "MonthlyCharges": 70.0,
        "TotalCharges": 210.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
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


def test_prediction_contains_retention_decision(
    service,
    customer,
):
    result = service.predict(customer)

    assert "retention_decision" in result

    decision = result[
        "retention_decision"
    ]

    assert decision[
        "decision_status"
    ] == "READY"


def test_retention_decision_contains_risk_information(
    service,
    customer,
):
    result = service.predict(customer)

    decision = result[
        "retention_decision"
    ]

    assert decision[
        "risk_tier"
    ] == result["risk_tier"]

    assert (
        decision["projected_churn"]
        == result["projected_churn"]
    )


def test_retention_decision_contains_evaluated_interventions(
    service,
    customer,
):
    result = service.predict(customer)

    decision = result[
        "retention_decision"
    ]

    assert (
        decision["evaluated_count"]
        == len(
            decision[
                "evaluated_interventions"
            ]
        )
    )


def test_simulation_layer_does_not_recurse(
    service,
    customer,
):
    result = service.predict(customer)

    assert result[
        "retention_decision"
    ]["decision_status"] == "READY"


def test_simulated_interventions_receive_roi(
    service,
    customer,
):
    result = service.predict(customer)

    interventions = result[
        "retention_decision"
    ][
        "evaluated_interventions"
    ]

    simulated = [
        item
        for item in interventions
        if item.get("roi") is not None
    ]

    assert len(simulated) >= 1

    for intervention in simulated:
        roi = intervention["roi"]

        assert (
            "estimated_net_value"
            in roi
        )

        assert (
            "economic_recommendation"
            in roi
        )


def test_existing_prediction_outputs_are_preserved(
    service,
    customer,
):
    result = service.predict(customer)

    required_fields = {
        "status",
        "current_tenure",
        "forecast_horizon",
        "target_month",
        "hazard_ratio",
        "projected_churn",
        "projected_retention",
        "risk_tier",
        "recommended_action",
        "forecast_curve",
        "explanation",
        "business_decision",
        "action_priority",
        "retention_interventions",
        "retention_decision",
    }

    assert required_fields.issubset(
        result.keys()
    )