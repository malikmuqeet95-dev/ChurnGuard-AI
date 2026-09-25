import pytest

from backend.src.intervention_library import (
    RetentionInterventionLibrary,
)


@pytest.fixture
def library():
    return RetentionInterventionLibrary()


def base_customer():
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


def test_month_to_month_customer_gets_contract_intervention(library):
    customer = base_customer()

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="CRITICAL",
        projected_churn=70.0,
    )

    ids = {
        candidate["intervention_id"]
        for candidate in candidates
    }

    assert "CONTRACT_CONVERSION" in ids


def test_electronic_check_customer_gets_payment_intervention(library):
    customer = base_customer()

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="MODERATE",
        projected_churn=40.0,
    )

    ids = {
        candidate["intervention_id"]
        for candidate in candidates
    }

    assert "PAYMENT_AUTOPAY" in ids


def test_missing_support_gets_support_intervention(library):
    customer = base_customer()

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="CRITICAL",
        projected_churn=60.0,
    )

    ids = {
        candidate["intervention_id"]
        for candidate in candidates
    }

    assert "TECH_SUPPORT_BUNDLE" in ids


def test_missing_security_gets_security_intervention(library):
    customer = base_customer()

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="MODERATE",
        projected_churn=35.0,
    )

    ids = {
        candidate["intervention_id"]
        for candidate in candidates
    }

    assert "ONLINE_SECURITY_BUNDLE" in ids


def test_early_customer_gets_onboarding_intervention(library):
    customer = base_customer()

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="MODERATE",
        projected_churn=40.0,
    )

    ids = {
        candidate["intervention_id"]
        for candidate in candidates
    }

    assert "ONBOARDING_SUPPORT" in ids


def test_long_term_healthy_customer_gets_loyalty_intervention(library):
    customer = base_customer()

    customer["tenure"] = 36
    customer["Contract"] = "Two year"
    customer["PaymentMethod"] = "Credit card (automatic)"
    customer["TechSupport"] = "Yes"
    customer["OnlineSecurity"] = "Yes"

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="HEALTHY",
        projected_churn=5.0,
    )

    ids = {
        candidate["intervention_id"]
        for candidate in candidates
    }

    assert "LOYALTY_RECOGNITION" in ids


def test_non_monthly_contract_does_not_get_conversion_intervention(
    library,
):
    customer = base_customer()
    customer["Contract"] = "Two year"

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="CRITICAL",
        projected_churn=60.0,
    )

    ids = {
        candidate["intervention_id"]
        for candidate in candidates
    }

    assert "CONTRACT_CONVERSION" not in ids


def test_no_internet_service_does_not_get_service_interventions(
    library,
):
    customer = base_customer()
    customer["InternetService"] = "No"

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="MODERATE",
        projected_churn=30.0,
    )

    ids = {
        candidate["intervention_id"]
        for candidate in candidates
    }

    assert "TECH_SUPPORT_BUNDLE" not in ids
    assert "ONLINE_SECURITY_BUNDLE" not in ids


def test_candidates_are_ranked_by_priority(library):
    customer = base_customer()

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="CRITICAL",
        projected_churn=70.0,
    )

    assert candidates

    ranks = [
        candidate["rank"]
        for candidate in candidates
    ]

    assert ranks == list(
        range(1, len(candidates) + 1)
    )


def test_intervention_contains_simulation_mapping(library):
    customer = base_customer()

    candidates = library.get_candidates(
        customer=customer,
        risk_tier="CRITICAL",
        projected_churn=70.0,
    )

    contract_candidate = next(
        candidate
        for candidate in candidates
        if candidate["intervention_id"]
        == "CONTRACT_CONVERSION"
    )

    assert contract_candidate[
        "simulation_changes"
    ] == {
        "Contract": "One year"
    }


def test_invalid_risk_tier_is_rejected(library):
    with pytest.raises(ValueError):
        library.get_candidates(
            customer=base_customer(),
            risk_tier="UNKNOWN",
            projected_churn=30.0,
        )


def test_invalid_churn_is_rejected(library):
    with pytest.raises(ValueError):
        library.get_candidates(
            customer=base_customer(),
            risk_tier="MODERATE",
            projected_churn=120.0,
        )