import pytest

from backend.app.prediction_service import (
    PredictionService,
)


def build_customer():

    return {
        "tenure": 3,
        "forecast_horizon": 6,

        "MonthlyCharges": 75.0,
        "TotalCharges": 225.0,

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


def test_intervention_simulator_returns_result():

    service = PredictionService()

    result = service.simulate_interventions(
        customer_data=build_customer(),
        scenarios=[
            {
                "name": "Annual Contract",
                "changes": {
                    "Contract": "One year",
                },
            }
        ],
    )

    assert result["status"] == "success"

    assert "baseline" in result

    assert "scenarios" in result

    assert result["scenario_count"] == 1


def test_intervention_contains_prediction_and_delta():

    service = PredictionService()

    result = service.simulate_interventions(
        customer_data=build_customer(),
        scenarios=[
            {
                "name": "Add Tech Support",
                "changes": {
                    "TechSupport": "Yes",
                },
            }
        ],
    )

    scenario = result["scenarios"][0]

    assert "prediction" in scenario

    assert "delta" in scenario

    assert "hazard_ratio" in (
        scenario["prediction"]
    )

    assert "projected_churn" in (
        scenario["prediction"]
    )

    assert "churn_change_pp" in (
        scenario["delta"]
    )


def test_baseline_customer_is_not_mutated():

    service = PredictionService()

    customer = build_customer()

    original_contract = customer["Contract"]

    service.simulate_interventions(
        customer_data=customer,
        scenarios=[
            {
                "name": "Annual Contract",
                "changes": {
                    "Contract": "One year",
                },
            }
        ],
    )

    assert (
        customer["Contract"]
        == original_contract
    )


def test_invalid_intervention_field_is_rejected():

    service = PredictionService()

    with pytest.raises(
        ValueError
    ):

        service.simulate_interventions(
            customer_data=build_customer(),
            scenarios=[
                {
                    "name": "Invalid Scenario",
                    "changes": {
                        "gender": "Female",
                    },
                }
            ],
        )


def test_invalid_category_value_is_rejected():

    service = PredictionService()

    with pytest.raises(
        ValueError
    ):

        service.simulate_interventions(
            customer_data=build_customer(),
            scenarios=[
                {
                    "name": "Bad Contract",
                    "changes": {
                        "Contract": "Five year",
                    },
                }
            ],
        )


def test_no_op_intervention_is_rejected():

    service = PredictionService()

    customer = build_customer()

    with pytest.raises(
        ValueError
    ):

        service.simulate_interventions(
            customer_data=customer,
            scenarios=[
                {
                    "name": "No Change",
                    "changes": {
                        "Contract":
                            "Month-to-month",
                    },
                }
            ],
        )


def test_multiple_scenarios_are_ranked():

    service = PredictionService()

    result = service.simulate_interventions(
        customer_data=build_customer(),
        scenarios=[
            {
                "name": "Annual Contract",
                "changes": {
                    "Contract": "One year",
                },
            },
            {
                "name": "Two Year Contract",
                "changes": {
                    "Contract": "Two year",
                },
            },
            {
                "name": "Add Tech Support",
                "changes": {
                    "TechSupport": "Yes",
                },
            },
        ],
    )

    scenarios = result["scenarios"]

    assert len(scenarios) == 3

    assert scenarios[0]["rank"] == 1
    assert scenarios[1]["rank"] == 2
    assert scenarios[2]["rank"] == 3

    churn_deltas = [
        scenario["delta"][
            "churn_change_pp"
        ]
        for scenario in scenarios
    ]

    assert churn_deltas == sorted(
        churn_deltas
    )


def test_delta_matches_prediction_difference():

    service = PredictionService()

    result = service.simulate_interventions(
        customer_data=build_customer(),
        scenarios=[
            {
                "name": "Annual Contract",
                "changes": {
                    "Contract": "One year",
                },
            }
        ],
    )

    baseline = result["baseline"]

    scenario = result["scenarios"][0]

    expected_delta = round(
        scenario["prediction"][
            "projected_churn"
        ]
        - baseline[
            "projected_churn"
        ],
        2,
    )

    assert (
        scenario["delta"][
            "churn_change_pp"
        ]
        == expected_delta
    )