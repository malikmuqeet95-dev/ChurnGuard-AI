from backend.app.prediction_service import PredictionService


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


def test_business_decision_is_returned():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    assert "business_decision" in result


def test_business_decision_has_required_fields():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    decision = result[
        "business_decision"
    ]

    assert "priority" in decision
    assert "strategy" in decision
    assert "objective" in decision
    assert "lifecycle_stage" in decision
    assert "primary_reason" in decision
    assert "business_signals" in decision
    assert "recommended_actions" in decision
    assert "action_count" in decision


def test_critical_customer_gets_urgent_priority():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    if result["risk_tier"] == "CRITICAL":

        decision = result[
            "business_decision"
        ]

        assert decision[
            "priority"
        ] == "URGENT"


def test_month_to_month_customer_gets_contract_signal():
    service = PredictionService()

    customer = build_customer()

    result = service.predict(
        customer
    )

    signals = result[
        "business_decision"
    ]["business_signals"]

    contract_signals = [
        signal
        for signal in signals
        if signal["category"] == "CONTRACT"
    ]

    assert len(
        contract_signals
    ) == 1

    assert (
        "Month-to-month"
        in contract_signals[0]["description"]
    )


def test_business_actions_are_not_empty():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    actions = result[
        "business_decision"
    ]["recommended_actions"]

    assert isinstance(
        actions,
        list,
    )

    assert len(actions) > 0


def test_lifecycle_stage_is_valid():
    service = PredictionService()

    result = service.predict(
        build_customer()
    )

    valid_stages = {
        "EARLY_LIFE",
        "DEVELOPING",
        "ESTABLISHED",
        "MATURE",
        "LONG_TERM",
    }

    assert (
        result[
            "business_decision"
        ]["lifecycle_stage"]
        in valid_stages
    )