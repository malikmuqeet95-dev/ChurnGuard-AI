import pytest

from backend.src.intervention_roi import (
    InterventionROIEngine,
)


@pytest.fixture
def engine():
    return InterventionROIEngine()


def test_positive_churn_improvement_generates_economic_value(
    engine,
):
    result = engine.calculate(
        intervention_id="CONTRACT_CONVERSION",
        churn_improvement_pp=10.0,
        customer_value=1200.0,
    )

    assert result["estimated_value_preserved"] > 0
    assert result["estimated_net_value"] > 0
    assert result["estimated_roi_pct"] > 0


def test_zero_improvement_produces_negative_net_value(
    engine,
):
    result = engine.calculate(
        intervention_id="CONTRACT_CONVERSION",
        churn_improvement_pp=0.0,
        customer_value=1200.0,
    )

    assert result["estimated_value_preserved"] == 0
    assert result["estimated_net_value"] < 0
    assert result["economic_recommendation"] == (
        "LOW_ECONOMIC_PRIORITY"
    )


def test_higher_churn_improvement_increases_value(
    engine,
):
    low = engine.calculate(
        intervention_id="TECH_SUPPORT_BUNDLE",
        churn_improvement_pp=5.0,
        customer_value=1200.0,
    )

    high = engine.calculate(
        intervention_id="TECH_SUPPORT_BUNDLE",
        churn_improvement_pp=15.0,
        customer_value=1200.0,
    )

    assert (
        high["estimated_net_value"]
        > low["estimated_net_value"]
    )


def test_customer_value_changes_economic_result(
    engine,
):
    low_value = engine.calculate(
        intervention_id="CONTRACT_CONVERSION",
        churn_improvement_pp=10.0,
        customer_value=500.0,
    )

    high_value = engine.calculate(
        intervention_id="CONTRACT_CONVERSION",
        churn_improvement_pp=10.0,
        customer_value=2000.0,
    )

    assert (
        high_value["estimated_net_value"]
        > low_value["estimated_net_value"]
    )


def test_unknown_intervention_uses_default_cost(
    engine,
):
    result = engine.calculate(
        intervention_id="UNKNOWN_INTERVENTION",
        churn_improvement_pp=10.0,
    )

    assert result["estimated_cost"] == 50.0


def test_negative_churn_improvement_is_rejected(
    engine,
):
    with pytest.raises(ValueError):
        engine.calculate(
            intervention_id="CONTRACT_CONVERSION",
            churn_improvement_pp=-5.0,
        )


def test_negative_customer_value_is_rejected(
    engine,
):
    with pytest.raises(ValueError):
        engine.calculate(
            intervention_id="CONTRACT_CONVERSION",
            churn_improvement_pp=5.0,
            customer_value=-100.0,
        )


def test_rank_orders_by_net_value(
    engine,
):
    simulations = [
        {
            "intervention_id": "CONTRACT_CONVERSION",
            "delta": {
                "churn_change_pp": -5.0,
            },
        },
        {
            "intervention_id": "PAYMENT_AUTOPAY",
            "delta": {
                "churn_change_pp": -10.0,
            },
        }    
    ]

    ranked = engine.rank(
        simulations,
        customer_value=1200.0,
    )

    assert len(ranked) == 2

    assert (
        ranked[0]["roi"]["estimated_net_value"]
        >= ranked[1]["roi"]["estimated_net_value"]
    )


def test_rank_rejects_empty_simulations(
    engine,
):
    with pytest.raises(ValueError):
        engine.rank([])


def test_rank_requires_intervention_id(
    engine,
):
    simulations = [
        {
            "delta": {
                "churn_change_pp": -5.0,
            },
        }
    ]

    with pytest.raises(ValueError):
        engine.rank(simulations)


def test_disclaimer_is_present(
    engine,
):
    result = engine.calculate(
        intervention_id="PAYMENT_AUTOPAY",
        churn_improvement_pp=5.0,
    )

    assert "causal" in result["disclaimer"].lower()