
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from backend.src.drift_policy import DriftPolicy


class RetrainingManager:
    """
    Creates retraining decision records.

    This class does not train or promote a model.
    It creates a structured recommendation for a later workflow.
    """

    def __init__(
        self,
        drift_policy: DriftPolicy | None = None,
    ) -> None:
        self.drift_policy = drift_policy or DriftPolicy()

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _quality_is_acceptable(
        model_quality: Dict[str, Any],
    ) -> bool:
        if not isinstance(model_quality, dict):
            return False

        quality_passed = model_quality.get("quality_passed")

        if quality_passed is not None:
            return bool(quality_passed)

        return True

    def create_retraining_record(
        self,
        drift_report: Dict[str, Any],
        model_quality: Dict[str, Any] | None = None,
        model_version: str = "unknown",
    ) -> Dict[str, Any]:
        model_quality = model_quality or {}

        policy_result = self.drift_policy.evaluate(drift_report)

        quality_acceptable = self._quality_is_acceptable(model_quality)

        drift_requires_retraining = policy_result[
            "decision"
        ] in {
            "RETRAINING_RECOMMENDED",
            "HUMAN_REVIEW_REQUIRED",
        }

        if not quality_acceptable:
            decision = "HUMAN_REVIEW_REQUIRED"
            reason = (
                "Current model quality information is unacceptable. "
                "Human review is required before retraining."
            )

        elif policy_result["decision"] == "HUMAN_REVIEW_REQUIRED":
            decision = "HUMAN_REVIEW_REQUIRED"
            reason = policy_result["reason"]

        elif drift_requires_retraining:
            decision = "RETRAINING_QUEUED"
            reason = policy_result["reason"]

        else:
            decision = "NO_RETRAINING_REQUIRED"
            reason = policy_result["reason"]

        return {
            "record_id": f"retrain-{uuid4().hex}",
            "created_at_utc": self._utc_timestamp(),
            "model_version": model_version,
            "decision": decision,
            "reason": reason,
            "retraining_requested": decision == "RETRAINING_QUEUED",
            "human_review_required": decision
            == "HUMAN_REVIEW_REQUIRED",
            "quality_acceptable": quality_acceptable,
            "drift_policy": policy_result,
            "model_quality": model_quality,
            "disclaimer": (
                "This record is an operational recommendation. "
                "It does not execute retraining or guarantee improved "
                "future model performance."
            ),
        }