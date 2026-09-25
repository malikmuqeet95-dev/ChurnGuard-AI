from __future__ import annotations

from typing import Any, Dict, List


class RetentionDecisionEngine:
    """
    Combines intervention simulations and economic estimates
    into one operational retention decision.

    This component does NOT modify the underlying ML prediction.

    It ranks candidate interventions using:
        1. modeled churn improvement
        2. estimated economic value
        3. operational priority

    IMPORTANT:
    The result is a decision-support recommendation.
    It is not evidence of causal treatment effectiveness.
    """

    PRIORITY_ORDER = {
        "URGENT": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4,
    }

    def _validate_inputs(
        self,
        risk_tier: str,
        projected_churn: float,
        action_priority: Dict[str, Any],
        interventions: List[Dict[str, Any]],
    ) -> None:
        if risk_tier not in {
            "CRITICAL",
            "MODERATE",
            "HEALTHY",
        }:
            raise ValueError(
                f"Unsupported risk tier: {risk_tier}"
            )

        try:
            projected_churn = float(projected_churn)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Projected churn must be numeric."
            ) from exc

        if not 0 <= projected_churn <= 100:
            raise ValueError(
                "Projected churn must be between 0 and 100."
            )

        if not isinstance(action_priority, dict):
            raise ValueError(
                "Action priority must be a dictionary."
            )

        if not isinstance(interventions, list):
            raise ValueError(
                "Interventions must be provided as a list."
            )

    def _get_operational_priority(
        self,
        risk_tier: str,
        intervention_priority: str,
    ) -> str:
        """
        Combine customer risk and intervention priority.

        The customer's risk tier establishes the upper urgency
        boundary while the intervention's own priority provides
        additional operational context.
        """

        risk_priority = {
            "CRITICAL": "URGENT",
            "MODERATE": "HIGH",
            "HEALTHY": "LOW",
        }

        customer_priority = risk_priority[risk_tier]

        customer_rank = self.PRIORITY_ORDER[
            customer_priority
        ]

        intervention_rank = self.PRIORITY_ORDER.get(
            intervention_priority,
            4,
        )

        # Choose the more urgent of the two.
        if customer_rank <= intervention_rank:
            return customer_priority

        return intervention_priority

    def _build_reason(
        self,
        risk_tier: str,
        projected_churn: float,
        selected: Dict[str, Any],
    ) -> str:
        intervention_name = selected.get(
            "name",
            "the selected intervention",
        )

        churn_improvement = selected.get(
            "roi",
            {},
        ).get(
            "modeled_churn_improvement_pp",
            0.0,
        )

        net_value = selected.get(
            "roi",
            {},
        ).get(
            "estimated_net_value",
            0.0,
        )

        return (
            f"The customer has {risk_tier.lower()} risk with "
            f"projected churn of {projected_churn:.2f}%. "
            f"{intervention_name} has the strongest available "
            f"combination of modeled churn improvement "
            f"({churn_improvement:.2f} percentage points) and "
            f"estimated net economic value "
            f"({net_value:.2f}) among the evaluated candidates."
        )

    def _recommendation_label(
        self,
        selected: Dict[str, Any] | None,
        risk_tier: str,
    ) -> str:
        if selected is None:
            if risk_tier == "CRITICAL":
                return "HUMAN_REVIEW_REQUIRED"

            if risk_tier == "MODERATE":
                return "MONITOR_AND_REVIEW"

            return "NO_TARGETED_INTERVENTION"

        economic_recommendation = selected.get(
            "roi",
            {},
        ).get(
            "economic_recommendation",
            "CONSIDER",
        )

        if economic_recommendation == "STRONG_CANDIDATE":
            return "RECOMMEND_INTERVENTION"

        if economic_recommendation == "CONSIDER":
            return "CONSIDER_INTERVENTION"

        return "REVIEW_ECONOMIC_PRIORITY"

    def _rank_candidates(
        self,
        interventions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rank interventions primarily by estimated net value,
        followed by modeled churn improvement.
        """

        ranked = sorted(
            interventions,
            key=lambda item: (
                float(
                    item.get(
                        "roi",
                        {},
                    ).get(
                        "estimated_net_value",
                        float("-inf"),
                    )
                ),
                float(
                    item.get(
                        "roi",
                        {},
                    ).get(
                        "modeled_churn_improvement_pp",
                        0.0,
                    )
                ),
            ),
            reverse=True,
        )

        for index, intervention in enumerate(
            ranked,
            start=1,
        ):
            intervention["decision_rank"] = index

        return ranked

    def decide(
        self,
        risk_tier: str,
        projected_churn: float,
        action_priority: Dict[str, Any],
        interventions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produce the final retention decision.

        The interventions supplied here should already contain
        simulation and ROI information.
        """

        self._validate_inputs(
            risk_tier=risk_tier,
            projected_churn=projected_churn,
            action_priority=action_priority,
            interventions=interventions,
        )

        projected_churn = float(projected_churn)

        ranked = self._rank_candidates(
            [
                dict(intervention)
                for intervention in interventions
            ]
        )

        selected = None

        for intervention in ranked:
            roi = intervention.get(
                "roi",
                {},
            )

            recommendation = roi.get(
                "economic_recommendation"
            )

            if recommendation in {
                "STRONG_CANDIDATE",
                "CONSIDER",
            }:
                selected = intervention
                break

        operational_priority = (
            action_priority.get(
                "priority"
            )
            or self._get_operational_priority(
                risk_tier=risk_tier,
                intervention_priority=(
                    selected.get("priority", "LOW")
                    if selected
                    else "LOW"
                ),
            )
        )

        if selected is not None:
            recommendation = (
                self._recommendation_label(
                    selected=selected,
                    risk_tier=risk_tier,
                )
            )

            reason = self._build_reason(
                risk_tier=risk_tier,
                projected_churn=projected_churn,
                selected=selected,
            )

            selected_intervention = {
                "intervention_id": selected.get(
                    "intervention_id"
                ),
                "name": selected.get(
                    "name"
                ),
                "category": selected.get(
                    "category"
                ),
                "priority": selected.get(
                    "priority"
                ),
                "channel": selected.get(
                    "channel"
                ),
                "recommended_action": selected.get(
                    "recommended_action"
                ),
                "simulation_changes": selected.get(
                    "simulation_changes",
                    {},
                ),
                "roi": selected.get(
                    "roi",
                    {},
                ),
            }

        else:
            recommendation = (
                self._recommendation_label(
                    selected=None,
                    risk_tier=risk_tier,
                )
            )

            selected_intervention = None

            reason = (
                f"The customer has {risk_tier.lower()} risk "
                f"with projected churn of "
                f"{projected_churn:.2f}%, but no evaluated "
                f"intervention currently meets the economic "
                f"selection criteria."
            )

        return {
            "decision_status": "READY",
            "risk_tier": risk_tier,
            "projected_churn": round(
                projected_churn,
                2,
            ),
            "operational_priority": operational_priority,
            "recommendation": recommendation,
            "reason": reason,
            "selected_intervention": (
                selected_intervention
            ),
            "evaluated_interventions": ranked,
            "evaluated_count": len(ranked),
            "decision_disclaimer": (
                "This is a decision-support recommendation "
                "based on model-based what-if simulations and "
                "configurable economic assumptions. It should "
                "not be interpreted as evidence that an "
                "intervention causally reduces churn."
            ),
        }