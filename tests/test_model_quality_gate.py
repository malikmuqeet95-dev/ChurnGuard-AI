
from __future__ import annotations

from backend.src.model_quality_gate import (
    evaluate_model_quality,
)


def valid_evaluation():
    return {
        "model_type": "CoxPHFitter",
        "training_c_index": 0.92,
        "validation_c_index": 0.85,
        "training_rows": 100,
        "validation_rows": 25,
        "feature_columns": [
            "feature_a",
            "feature_b",
        ],
    }


def test_quality_gate_passes_valid_evaluation():
    result = evaluate_model_quality(
        valid_evaluation()
    )

    assert result.passed is True
    assert all(
        check.passed
        for check in result.checks
    )


def test_quality_gate_rejects_low_validation_score():
    evaluation = valid_evaluation()
    evaluation["validation_c_index"] = 0.55

    result = evaluate_model_quality(
        evaluation
    )

    assert result.passed is False

    failed_names = {
        check.name
        for check in result.checks
        if not check.passed
    }

    assert (
        "minimum_validation_c_index"
        in failed_names
    )


def test_quality_gate_rejects_large_overfitting_gap():
    evaluation = valid_evaluation()
    evaluation["training_c_index"] = 0.99
    evaluation["validation_c_index"] = 0.60

    result = evaluate_model_quality(
        evaluation
    )

    assert result.passed is False

    failed_names = {
        check.name
        for check in result.checks
        if not check.passed
    }

    assert (
        "overfitting_gap"
        in failed_names
    )


def test_quality_gate_rejects_invalid_model_type():
    evaluation = valid_evaluation()
    evaluation["model_type"] = "InvalidModel"

    result = evaluate_model_quality(
        evaluation
    )

    assert result.passed is False

    failed_names = {
        check.name
        for check in result.checks
        if not check.passed
    }

    assert (
        "model_type"
        in failed_names
    )


def test_quality_gate_rejects_empty_features():
    evaluation = valid_evaluation()
    evaluation["feature_columns"] = []

    result = evaluate_model_quality(
        evaluation
    )

    assert result.passed is False

    failed_names = {
        check.name
        for check in result.checks
        if not check.passed
    }

    assert (
        "feature_columns"
        in failed_names
    )