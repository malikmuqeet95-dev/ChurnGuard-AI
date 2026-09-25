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


def test_action_priority_is_returned():

    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    assert "action_priority" in result


def test_action_priority_has_required_fields():

    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    priority = result[
        "action_priority"
    ]

    assert "priority" in priority
    assert "priority_rank" in priority
    assert "action_window" in priority
    assert "workflow" in priority
    assert "reason" in priority
    assert "recommended_actions" in priority
    assert "action_count" in priority


def test_critical_customer_gets_urgent_action():

    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    if result["risk_tier"] == "CRITICAL":

        priority = result[
            "action_priority"
        ]

        assert (
            priority["priority"]
            == "URGENT"
        )

        assert (
            priority["action_window"]
            == "WITHIN_24_HOURS"
        )


def test_moderate_customer_gets_high_priority():

    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    if result["risk_tier"] == "MODERATE":

        priority = result[
            "action_priority"
        ]

        assert (
            priority["priority"]
            == "HIGH"
        )

        assert (
            priority["action_window"]
            == "WITHIN_3_DAYS"
        )


def test_action_priority_contains_workflow():

    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    priority = result[
        "action_priority"
    ]

    valid_workflows = {
        "HUMAN_RETENTION_OUTREACH",
        "CUSTOMER_SUCCESS_FOLLOW_UP",
        "AUTOMATED_ENGAGEMENT",
        "NORMAL_MONITORING",
    }

    assert (
        priority["workflow"]
        in valid_workflows
    )


def test_action_priority_has_actions():

    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    priority = result[
        "action_priority"
    ]

    assert isinstance(
        priority[
            "recommended_actions"
        ],
        list,
    )

    assert (
        len(
            priority[
                "recommended_actions"
            ]
        )
        > 0
    )