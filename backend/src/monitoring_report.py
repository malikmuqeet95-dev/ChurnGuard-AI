
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4


class MonitoringReportBuilder:
    """
    Combines drift, retraining, and performance assessments
    into a single monitoring report.

    This component does not execute operational actions.
    """

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _status_rank(status: str) -> int:
        return {
            "HEALTHY": 0,
            "LOW": 0,
            "WARNING": 1,
            "CRITICAL": 2,
        }.get(str(status).upper(), 2)

    def build(
        self,
        drift_policy_result: Dict[str, Any],
        retraining_result: Dict[str, Any],
        performance_result: Dict[str, Any],
        model_version: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Build a unified monitoring report.

        Each input must be a dictionary produced by the
        corresponding monitoring component.
        """

        inputs = {
            "drift_policy": drift_policy_result,
            "retraining": retraining_result,
            "performance": performance_result,
        }

        invalid_sections = [
            section_name
            for section_name, section_value in inputs.items()
            if not isinstance(section_value, dict)
        ]

        if invalid_sections:
            return {
                "report_id": f"monitoring-invalid-{uuid4().hex}",
                "created_at_utc": self._utc_timestamp(),
                "model_version": model_version,
                "overall_status": "CRITICAL",
                "decision": "HUMAN_REVIEW_REQUIRED",
                "invalid_sections": invalid_sections,
                "disclaimer": (
                    "This report contains invalid monitoring sections "
                    "and requires human review."
                ),
            }

        statuses = [
            drift_policy_result.get("severity", "CRITICAL"),
            performance_result.get("status", "CRITICAL"),
        ]

        normalized_statuses = [
            self._normalize_status(status)
            for status in statuses
        ]

        highest_status = max(
            normalized_statuses,
            key=self._status_rank,
        )

        human_review_required = (
            drift_policy_result.get("decision")
            == "HUMAN_REVIEW_REQUIRED"
            or retraining_result.get("decision")
            == "HUMAN_REVIEW_REQUIRED"
            or performance_result.get("decision")
            == "HUMAN_REVIEW_REQUIRED"
        )

        retraining_requested = bool(
            retraining_result.get("retraining_requested", False)
        )

        if human_review_required or highest_status == "CRITICAL":
            overall_status = "CRITICAL"
            decision = "HUMAN_REVIEW_REQUIRED"

        elif retraining_requested:
            overall_status = "WARNING"
            decision = "RETRAINING_QUEUED"

        elif highest_status == "WARNING":
            overall_status = "WARNING"
            decision = "MONITOR_CLOSELY"

        else:
            overall_status = "HEALTHY"
            decision = "NO_ACTION"

        return {
            "report_id": f"monitoring-{uuid4().hex}",
            "created_at_utc": self._utc_timestamp(),
            "model_version": model_version,
            "overall_status": overall_status,
            "decision": decision,
            "human_review_required": human_review_required,
            "retraining_requested": retraining_requested,
            "sections": inputs,
            "disclaimer": (
                "This monitoring report is decision support. "
                "It does not execute retraining, deployment, "
                "or production model replacement."
            ),
        }

    @staticmethod
    def _normalize_status(status: Any) -> str:
        normalized = str(status).upper()

        if normalized in {"HEALTHY", "LOW"}:
            return "HEALTHY"

        if normalized in {"WARNING", "MODERATE"}:
            return "WARNING"

        return "CRITICAL"