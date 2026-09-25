
from __future__ import annotations

from backend.src.model_promotion import (
    evaluate_promotion_eligibility,
)


def evaluation(
    validation_c_index: float,
):
    return {
        "model_type": "CoxPHFitter",
        "validation_c_index": validation_c_index,
        "feature_columns": [
            "SeniorCitizen",
            "MonthlyCharges_scaled",
            "TotalCharges_scaled",
        ],
    }


def test_candidate_is_eligible_when_quality_is_acceptable():
    candidate = evaluation(0.85)
    production = evaluation(0.84)

    result = evaluate_promotion_eligibility(
        candidate_evaluation=candidate,
        production_evaluation=production,
    )

    assert result.eligible is True
    assert result.status == (
        "ELIGIBLE_FOR_REVIEW"
    )


def test_candidate_is_rejected_when_score_is_too_low():
    candidate = evaluation(0.65)
    production = evaluation(0.84)

    result = evaluate_promotion_eligibility(
        candidate_evaluation=candidate,
        production_evaluation=production,
    )

    assert result.eligible is False
    assert result.status == "REJECTED"


def test_candidate_is_rejected_for_large_degradation():
    candidate = evaluation(0.80)
    production = evaluation(0.85)

    result = evaluate_promotion_eligibility(
        candidate_evaluation=candidate,
        production_evaluation=production,
    )

    assert result.eligible is False

    failed_checks = {
        check.name
        for check in result.checks
        if not check.passed
    }

    assert (
        "performance_degradation"
        in failed_checks
    )


def test_candidate_is_rejected_for_feature_mismatch():
    candidate = evaluation(0.86)
    production = evaluation(0.84)

    candidate["feature_columns"] = [
        "different_feature"
    ]

    result = evaluate_promotion_eligibility(
        candidate_evaluation=candidate,
        production_evaluation=production,
    )

    assert result.eligible is False

    failed_checks = {
        check.name
        for check in result.checks
        if not check.passed
    }

    assert (
        "feature_schema_compatibility"
        in failed_checks
    )


def test_promotion_does_not_change_evaluation_data():
    candidate = evaluation(0.85)
    production = evaluation(0.84)

    original_candidate = candidate.copy()
    original_production = production.copy()

    evaluate_promotion_eligibility(
        candidate_evaluation=candidate,
        production_evaluation=production,
    )

    assert candidate == original_candidate
    assert production == original_production