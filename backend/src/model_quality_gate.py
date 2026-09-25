
"""
Automated quality checks for trained survival models.

A quality gate produces a pass/fail result. It does not automatically
promote a model to production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QualityCheck:
    """One quality check result."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class ModelQualityGateResult:
    """Complete quality gate result."""

    passed: bool
    checks: tuple[QualityCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-compatible dictionary."""

        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "message": check.message,
                }
                for check in self.checks
            ],
        }


def evaluate_model_quality(
    evaluation: dict[str, Any],
    minimum_validation_c_index: float = 0.70,
    maximum_overfitting_gap: float = 0.20,
) -> ModelQualityGateResult:
    """
    Evaluate model quality using evaluation metadata.

    The validation C-index threshold is configurable.
    """

    checks: list[QualityCheck] = []

    model_type = evaluation.get(
        "model_type"
    )

    checks.append(
        QualityCheck(
            name="model_type",
            passed=model_type == "CoxPHFitter",
            message=(
                "Model type is CoxPHFitter"
                if model_type == "CoxPHFitter"
                else (
                    "Unexpected model type: "
                    f"{model_type}"
                )
            ),
        )
    )

    training_c_index = float(
        evaluation.get(
            "training_c_index",
            -1,
        )
    )

    validation_c_index = float(
        evaluation.get(
            "validation_c_index",
            -1,
        )
    )

    checks.append(
        QualityCheck(
            name="training_c_index_range",
            passed=0.0 <= training_c_index <= 1.0,
            message=(
                f"Training C-index: {training_c_index}"
            ),
        )
    )

    checks.append(
        QualityCheck(
            name="validation_c_index_range",
            passed=0.0 <= validation_c_index <= 1.0,
            message=(
                f"Validation C-index: {validation_c_index}"
            ),
        )
    )

    checks.append(
        QualityCheck(
            name="minimum_validation_c_index",
            passed=(
                validation_c_index
                >= minimum_validation_c_index
            ),
            message=(
                "Validation C-index meets minimum "
                f"threshold: {minimum_validation_c_index}"
            ),
        )
    )

    training_rows = int(
        evaluation.get(
            "training_rows",
            0,
        )
    )

    validation_rows = int(
        evaluation.get(
            "validation_rows",
            0,
        )
    )

    checks.append(
        QualityCheck(
            name="training_rows",
            passed=training_rows > 0,
            message=f"Training rows: {training_rows}",
        )
    )

    checks.append(
        QualityCheck(
            name="validation_rows",
            passed=validation_rows > 0,
            message=(
                f"Validation rows: {validation_rows}"
            ),
        )
    )

    feature_columns = evaluation.get(
        "feature_columns",
        [],
    )

    checks.append(
        QualityCheck(
            name="feature_columns",
            passed=bool(feature_columns),
            message=(
                f"Feature count: {len(feature_columns)}"
            ),
        )
    )

    overfitting_gap = (
        training_c_index
        - validation_c_index
    )

    checks.append(
        QualityCheck(
            name="overfitting_gap",
            passed=(
                overfitting_gap
                <= maximum_overfitting_gap
            ),
            message=(
                "Training-validation C-index gap: "
                f"{overfitting_gap:.4f}; "
                "maximum allowed: "
                f"{maximum_overfitting_gap:.4f}"
            ),
        )
    )

    all_passed = all(
        check.passed
        for check in checks
    )

    return ModelQualityGateResult(
        passed=all_passed,
        checks=tuple(checks),
    )