import pytest

from backend.src.retention_decision import (
    RetentionDecisionEngine,
)


@pytest.fixture
def engine():
    return RetentionDecisionEngine()


def make_intervention(
    intervention_id,
    name,
    net_value,
    improvement,
    priority="HIGH",
):
    return {
        "intervention_id": intervention_id,
        "name": name,
        "category": "TEST",
        "priority": priority,
        "channel": "CUSTOMER_SUCCESS",
        "recommended_action": (
            f"Perform {name}"
        ),
        "simulation_changes": {},
        "roi": {
            "modeled_churn_improvement_pp": improvement,
            "estimated_net_value": net_value,
            "estimated_roi_pct": (
                net_value / 50 * 100
            ),
            "economic_recommendation": (
                "STRONG_CANDIDATE"
                if net_value > 100
                else "CONSIDER"
                if net_value > 0
                else "LOW_ECONOMIC_PRIORITY"
            ),
        },
    }


def test_engine_selects_best_economic_intervention(
    engine,
):
    interventions = [
        make_intervention(
            "A",
            "Contract Conversion",
            net_value=250,
            improvement=12,
        ),
        make_intervention(
            "B",
            "Support Offer",
            net_value=100,
            improvement=8,
        ),
    ]

    result = engine.decide(
        risk_tier="CRITICAL",
        projected_churn=65.0,
        action_priority={
            "priority": "URGENT",
        },
        interventions=interventions,
    )

    assert (
        result["selected_intervention"]
        ["intervention_id"]
        == "A"
    )


def test_recommendation_is_generated(
    engine,
):
    interventions = [
        make_intervention(
            "A",
            "Contract Conversion",
            net_value=250,
            improvement=12,
        )
    ]

    result = engine.decide(
        risk_tier="CRITICAL",
        projected_churn=65.0,
        action_priority={
            "priority": "URGENT",
        },
        interventions=interventions,
    )

    assert result["recommendation"] == (
        "RECOMMEND_INTERVENTION"
    )


def test_urgent_priority_is_preserved(
    engine,
):
    interventions = [
        make_intervention(
            "A",
            "Contract Conversion",
            net_value=250,
            improvement=12,
        )
    ]

    result = engine.decide(
        risk_tier="CRITICAL",
        projected_churn=65.0,
        action_priority={
            "priority": "URGENT",
        },
        interventions=interventions,
    )

    assert result[
        "operational_priority"
    ] == "URGENT"


def test_candidates_are_ranked(
    engine,
):
    interventions = [
        make_intervention(
            "A",
            "Intervention A",
            net_value=50,
            improvement=5,
        ),
        make_intervention(
            "B",
            "Intervention B",
            net_value=200,
            improvement=7,
        ),
        make_intervention(
            "C",
            "Intervention C",
            net_value=100,
            improvement=6,
        ),
    ]

    result = engine.decide(
        risk_tier="MODERATE",
        projected_churn=40.0,
        action_priority={
            "priority": "HIGH",
        },
        interventions=interventions,
    )

    ranked_ids = [
        item["intervention_id"]
        for item in result[
            "evaluated_interventions"
        ]
    ]

    assert ranked_ids == [
        "B",
        "C",
        "A",
    ]


def test_no_economic_candidate_requires_review(
    engine,
):
    interventions = [
        make_intervention(
            "A",
            "Expensive Intervention",
            net_value=-50,
            improvement=3,
        )
    ]

    result = engine.decide(
        risk_tier="CRITICAL",
        projected_churn=65.0,
        action_priority={
            "priority": "URGENT",
        },
        interventions=interventions,
    )

    assert (
        result["selected_intervention"]
        is None
    )

    assert result[
        "recommendation"
    ] == "HUMAN_REVIEW_REQUIRED"


def test_healthy_customer_without_candidate(
    engine,
):
    result = engine.decide(
        risk_tier="HEALTHY",
        projected_churn=5.0,
        action_priority={
            "priority": "LOW",
        },
        interventions=[],
    )

    assert (
        result["selected_intervention"]
        is None
    )

    assert result[
        "recommendation"
    ] == "NO_TARGETED_INTERVENTION"


def test_moderate_customer_without_candidate(
    engine,
):
    result = engine.decide(
        risk_tier="MODERATE",
        projected_churn=30.0,
        action_priority={
            "priority": "HIGH",
        },
        interventions=[],
    )

    assert (
        result["recommendation"]
        == "MONITOR_AND_REVIEW"
    )


def test_invalid_risk_tier_is_rejected(
    engine,
):
    with pytest.raises(ValueError):
        engine.decide(
            risk_tier="UNKNOWN",
            projected_churn=30.0,
            action_priority={
                "priority": "HIGH",
            },
            interventions=[],
        )


def test_invalid_churn_is_rejected(
    engine,
):
    with pytest.raises(ValueError):
        engine.decide(
            risk_tier="MODERATE",
            projected_churn=150.0,
            action_priority={
                "priority": "HIGH",
            },
            interventions=[],
        )


def test_invalid_action_priority_is_rejected(
    engine,
):
    with pytest.raises(ValueError):
        engine.decide(
            risk_tier="MODERATE",
            projected_churn=30.0,
            action_priority="HIGH",
            interventions=[],
        )


def test_invalid_interventions_is_rejected(
    engine,
):
    with pytest.raises(ValueError):
        engine.decide(
            risk_tier="MODERATE",
            projected_churn=30.0,
            action_priority={
                "priority": "HIGH",
            },
            interventions="invalid",
        )


def test_reason_contains_churn_and_intervention(
    engine,
):
    interventions = [
        make_intervention(
            "A",
            "Contract Conversion",
            net_value=250,
            improvement=12,
        )
    ]

    result = engine.decide(
        risk_tier="CRITICAL",
        projected_churn=65.0,
        action_priority={
            "priority": "URGENT",
        },
        interventions=interventions,
    )

    assert "65.00%" in result["reason"]
    assert "Contract Conversion" in result["reason"]


def test_disclaimer_is_present(
    engine,
):
    result = engine.decide(
        risk_tier="HEALTHY",
        projected_churn=5.0,
        action_priority={
            "priority": "LOW",
        },
        interventions=[],
    )

    assert "causal" in (
        result["decision_disclaimer"]
        .lower()
    )