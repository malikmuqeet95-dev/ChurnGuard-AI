"""
backend/src/drift_policy.py
Converts drift-monitoring results into an operational policy decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class DriftPolicyConfig:
    warning_psi: float = 0.10
    critical_psi: float = 0.25
    warning_feature_ratio: float = 0.20
    critical_feature_ratio: float = 0.50
    warning_missing_rate: float = 0.05
    critical_missing_rate: float = 0.20


class DriftPolicy:
    """
    Converts drift-monitoring results into an operational policy decision.

    This policy does not retrain models automatically.
    It produces a recommendation for monitoring or human review.
    """

    def __init__(
        self,
        config: DriftPolicyConfig | None = None,
    ) -> None:
        self.config = config or DriftPolicyConfig()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_features(self, drift_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]] | None:
        """Extracts normalized feature dict from either format."""
        if "features" in drift_report:
            features = drift_report["features"]
            return features if isinstance(features, dict) else None

        if "results" in drift_report and isinstance(drift_report["results"], (list, tuple)):
            normalized = {}
            for item in drift_report["results"]:
                if isinstance(item, dict) and "feature_name" in item:
                    normalized[item["feature_name"]] = {
                        "psi": item.get("psi", 0.0),
                        "missing_rate": item.get("current_missing_rate", item.get("missing_rate", 0.0)),
                    }
            return normalized

        return None

    def evaluate(
        self,
        drift_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a drift-monitoring report."""
        features = self._normalize_features(drift_report)

        if features is None:
            return {
                "decision": "HUMAN_REVIEW_REQUIRED",
                "severity": "CRITICAL",
                "retraining_recommended": False,
                "reason": "Invalid drift report feature structure.",
                "drifted_features": [],
                "critical_features": [],
            }

        total_features = len(features)
        warning_features: List[str] = []
        critical_features: List[str] = []

        for feature_name, feature_result in features.items():
            if not isinstance(feature_result, dict):
                critical_features.append(str(feature_name))
                continue

            psi = self._safe_float(feature_result.get("psi"))
            missing_rate = self._safe_float(
                feature_result.get("missing_rate", feature_result.get("current_missing_rate", 0.0))
            )

            if (
                psi >= self.config.critical_psi
                or missing_rate >= self.config.critical_missing_rate
            ):
                critical_features.append(str(feature_name))

            elif (
                psi >= self.config.warning_psi
                or missing_rate >= self.config.warning_missing_rate
            ):
                warning_features.append(str(feature_name))

        critical_ratio = (
            len(critical_features) / total_features
            if total_features
            else 0.0
        )

        warning_ratio = (
            len(warning_features) / total_features
            if total_features
            else 0.0
        )

        if not features:
            decision = "NO_ACTION"
            severity = "LOW"
            reason = "No feature drift results were available."

        elif critical_features or critical_ratio >= self.config.critical_feature_ratio:
            decision = "HUMAN_REVIEW_REQUIRED"
            severity = "CRITICAL"
            reason = "Critical drift or missing-value conditions were detected."

        elif (
            warning_ratio >= self.config.warning_feature_ratio
            or warning_features
        ):
            decision = "RETRAINING_RECOMMENDED"
            severity = "WARNING"
            reason = (
                "Feature distribution changes require model performance review "
                "and possible retraining."
            )

        else:
            decision = "NO_ACTION"
            severity = "LOW"
            reason = "Drift remains below configured alert thresholds."

        return {
            "decision": decision,
            "severity": severity,
            "retraining_recommended": decision
            in {
                "RETRAINING_RECOMMENDED",
                "HUMAN_REVIEW_REQUIRED",
            },
            "reason": reason,
            "drifted_features": sorted(
                set(warning_features + critical_features)
            ),
            "critical_features": sorted(set(critical_features)),
            "warning_features": sorted(set(warning_features)),
            "total_features": total_features,
            "warning_feature_ratio": round(warning_ratio, 4),
            "critical_feature_ratio": round(critical_ratio, 4),
            "policy_config": asdict(self.config),
        }