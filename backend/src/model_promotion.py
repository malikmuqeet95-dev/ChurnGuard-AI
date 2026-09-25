
"""
Candidate model approval and promotion eligibility.

This module compares evaluation metadata. It does not copy, replace,
or activate production model artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionCheck:
    """A single candidate promotion check."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class PromotionDecision:
    """Decision about candidate model eligibility."""

    eligible: bool
    status: str
    checks: tuple[PromotionCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the decision into a JSON-compatible dictionary."""

        return {
            "eligible": self.eligible,
            "status": self.status,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "message": check.message,
                }
                for check in self.checks
            ],
        }


def evaluate_promotion_eligibility(
    candidate_evaluation: dict[str, Any],
    production_evaluation: dict[str, Any],
    minimum_validation_c_index: float = 0.70,
    maximum_allowed_degradation: float = 0.02,
) -> PromotionDecision:
    """
    Compare a candidate model with a production baseline.

    The function only evaluates eligibility. It does not promote a model.
    """

    checks: list[PromotionCheck] = []

    candidate_model_type = candidate_evaluation.get(
        "model_type"
    )

    production_model_type = production_evaluation.get(
        "model_type"
    )

    checks.append(
        PromotionCheck(
            name="model_type_compatibility",
            passed=(
                candidate_model_type == "CoxPHFitter"
                and production_model_type == "CoxPHFitter"
            ),
            message=(
                "Candidate and production models use "
                "CoxPHFitter"
                if (
                    candidate_model_type == "CoxPHFitter"
                    and production_model_type == "CoxPHFitter"
                )
                else (
                    "Model type mismatch or unsupported "
                    f"candidate={candidate_model_type}, "
                    f"production={production_model_type}"
                )
            ),
        )
    )

    candidate_features = tuple(
        candidate_evaluation.get(
            "feature_columns",
            [],
        )
    )

    production_features = tuple(
        production_evaluation.get(
            "feature_columns",
            [],
        )
    )

    checks.append(
        PromotionCheck(
            name="feature_schema_compatibility",
            passed=(
                bool(candidate_features)
                and candidate_features == production_features
            ),
            message=(
                "Candidate feature schema matches production"
                if (
                    bool(candidate_features)
                    and candidate_features == production_features
                )
                else (
                    "Candidate feature schema does not "
                    "match production"
                )
            ),
        )
    )

    candidate_score = float(
        candidate_evaluation.get(
            "validation_c_index",
            -1,
        )
    )

    production_score = float(
        production_evaluation.get(
            "validation_c_index",
            -1,
        )
    )

    checks.append(
        PromotionCheck(
            name="candidate_minimum_validation_score",
            passed=(
                0.0 <= candidate_score <= 1.0
                and candidate_score
                >= minimum_validation_c_index
            ),
            message=(
                f"Candidate validation C-index: "
                f"{candidate_score:.4f}; "
                f"minimum: "
                f"{minimum_validation_c_index:.4f}"
            ),
        )
    )

    degradation = (
        production_score - candidate_score
    )

    checks.append(
        PromotionCheck(
            name="performance_degradation",
            passed=(
                0.0 <= production_score <= 1.0
                and 0.0 <= candidate_score <= 1.0
                and degradation
                <= maximum_allowed_degradation
            ),
            message=(
                f"Production score: {production_score:.4f}; "
                f"candidate score: {candidate_score:.4f}; "
                f"degradation: {degradation:.4f}; "
                f"maximum allowed: "
                f"{maximum_allowed_degradation:.4f}"
            ),
        )
    )

    eligible = all(
        check.passed
        for check in checks
    )

    return PromotionDecision(
        eligible=eligible,
        status=(
            "ELIGIBLE_FOR_REVIEW"
            if eligible
            else "REJECTED"
        ),
        checks=tuple(checks),
    )