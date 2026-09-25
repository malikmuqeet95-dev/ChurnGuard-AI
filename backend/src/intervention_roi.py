from __future__ import annotations

from typing import Any, Dict, List


class InterventionROIEngine:
    """
    Estimates business efficiency for model-based retention
    intervention scenarios.

    IMPORTANT:
    These are planning estimates, NOT causal measurements.

    The engine combines:
      - modeled churn improvement
      - estimated intervention cost
      - estimated customer value

    It should be used for prioritization and scenario planning,
    not as proof that an intervention causes retention.
    """

    DEFAULT_CUSTOMER_VALUE = 1200.0

    INTERVENTION_COSTS = {
        "CONTRACT_CONVERSION": 80.0,
        "PAYMENT_AUTOPAY": 15.0,
        "TECH_SUPPORT_BUNDLE": 60.0,
        "ONLINE_SECURITY_BUNDLE": 50.0,
        "ONBOARDING_SUPPORT": 40.0,
        "LOYALTY_RECOGNITION": 75.0,
    }

    INTERVENTION_VALUE_WEIGHTS = {
        "CONTRACT_CONVERSION": 1.00,
        "PAYMENT_AUTOPAY": 0.70,
        "TECH_SUPPORT_BUNDLE": 0.80,
        "ONLINE_SECURITY_BUNDLE": 0.70,
        "ONBOARDING_SUPPORT": 0.90,
        "LOYALTY_RECOGNITION": 0.60,
    }

    def _validate_numeric(
        self,
        value: Any,
        field_name: str,
    ) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be numeric."
            ) from exc

        if value < 0:
            raise ValueError(
                f"{field_name} cannot be negative."
            )

        return value

    def _get_intervention_cost(
        self,
        intervention_id: str,
    ) -> float:
        return float(
            self.INTERVENTION_COSTS.get(
                intervention_id,
                50.0,
            )
        )

    def _get_value_weight(
        self,
        intervention_id: str,
    ) -> float:
        return float(
            self.INTERVENTION_VALUE_WEIGHTS.get(
                intervention_id,
                0.75,
            )
        )

    def calculate(
        self,
        intervention_id: str,
        churn_improvement_pp: float,
        customer_value: float | None = None,
    ) -> Dict[str, Any]:
        """
        Calculate estimated intervention efficiency.

        churn_improvement_pp:
            Positive number means modeled churn reduction.

        Example:
            baseline churn = 50%
            scenario churn = 40%

            improvement = 10 percentage points.
        """

        churn_improvement_pp = self._validate_numeric(
            churn_improvement_pp,
            "Churn improvement",
        )

        if customer_value is None:
            customer_value = self.DEFAULT_CUSTOMER_VALUE
        else:
            customer_value = self._validate_numeric(
                customer_value,
                "Customer value",
            )

        cost = self._get_intervention_cost(
            intervention_id
        )

        value_weight = self._get_value_weight(
            intervention_id
        )

        normalized_improvement = (
            churn_improvement_pp / 100.0
        )

        estimated_value_preserved = (
            normalized_improvement
            * customer_value
            * value_weight
        )

        estimated_net_value = (
            estimated_value_preserved - cost
        )

        if cost > 0:
            roi_percent = (
                estimated_net_value / cost
            ) * 100.0
        else:
            roi_percent = 0.0

        if estimated_net_value > 100:
            recommendation = "STRONG_CANDIDATE"
        elif estimated_net_value > 0:
            recommendation = "CONSIDER"
        else:
            recommendation = "LOW_ECONOMIC_PRIORITY"

        return {
            "intervention_id": intervention_id,
            "estimated_cost": round(cost, 2),
            "customer_value_assumption": round(
                customer_value,
                2,
            ),
            "modeled_churn_improvement_pp": round(
                churn_improvement_pp,
                2,
            ),
            "estimated_value_preserved": round(
                estimated_value_preserved,
                2,
            ),
            "estimated_net_value": round(
                estimated_net_value,
                2,
            ),
            "estimated_roi_pct": round(
                roi_percent,
                2,
            ),
            "economic_recommendation": recommendation,
            "disclaimer": (
                "Economic impact is an estimated planning "
                "metric based on model output and configurable "
                "business assumptions. It is not a causal "
                "treatment-effect estimate."
            ),
        }

    def rank(
        self,
        simulations: List[Dict[str, Any]],
        customer_value: float | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank completed intervention simulations by estimated
        net economic value.
        """

        if not isinstance(simulations, list):
            raise ValueError(
                "Simulations must be provided as a list."
            )

        if not simulations:
            raise ValueError(
                "At least one simulation is required."
            )

        results = []

        for simulation in simulations:
            if not isinstance(simulation, dict):
                raise ValueError(
                    "Each simulation must be a dictionary."
                )

            intervention_id = simulation.get(
                "intervention_id"
            )

            if not intervention_id:
                raise ValueError(
                    "Simulation is missing intervention_id."
                )

            delta = simulation.get(
                "delta",
                {},
            )

            if not isinstance(delta, dict):
                raise ValueError(
                    "Simulation delta must be a dictionary."
                )

            churn_change = float(
                delta.get(
                    "churn_change_pp",
                    0.0,
                )
            )

            # Negative churn_change means improvement.
            improvement = max(
                0.0,
                -churn_change,
            )

            roi_result = self.calculate(
                intervention_id=intervention_id,
                churn_improvement_pp=improvement,
                customer_value=customer_value,
            )

            enriched = dict(simulation)
            enriched["roi"] = roi_result

            results.append(enriched)

        results.sort(
            key=lambda item: (
                item["roi"]["estimated_net_value"],
                item["roi"]["estimated_roi_pct"],
            ),
            reverse=True,
        )

        for index, item in enumerate(
            results,
            start=1,
        ):
            item["economic_rank"] = index

        return results