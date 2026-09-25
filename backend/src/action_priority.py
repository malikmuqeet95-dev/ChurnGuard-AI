from __future__ import annotations

from typing import Any, Dict, List


class ActionPriorityEngine:
    """
    Converts model and business outputs into an operational
    retention priority.

    This layer does NOT change the model prediction.

    It translates prediction + business context into
    an operational workflow recommendation.
    """

    # ============================================================
    # PRIORITY ORDER
    # ============================================================

    PRIORITY_ORDER = {
        "URGENT": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4,
    }

    # ============================================================
    # RISK TIER → DEFAULT PRIORITY
    # ============================================================

    RISK_PRIORITY = {
        "CRITICAL": "URGENT",
        "MODERATE": "HIGH",
        "HEALTHY": "LOW",
    }

    # ============================================================
    # TIME WINDOWS
    # ============================================================

    ACTION_WINDOWS = {
        "URGENT": "WITHIN_24_HOURS",
        "HIGH": "WITHIN_3_DAYS",
        "MEDIUM": "WITHIN_7_DAYS",
        "LOW": "WITHIN_30_DAYS",
    }

    # ============================================================
    # WORKFLOWS
    # ============================================================

    WORKFLOWS = {
        "URGENT": "HUMAN_RETENTION_OUTREACH",
        "HIGH": "CUSTOMER_SUCCESS_FOLLOW_UP",
        "MEDIUM": "AUTOMATED_ENGAGEMENT",
        "LOW": "NORMAL_MONITORING",
    }

    # ============================================================
    # PRIORITY CALCULATION
    # ============================================================

    def _base_priority(
        self,
        risk_tier: str,
    ) -> str:

        if risk_tier not in self.RISK_PRIORITY:
            raise ValueError(
                f"Unsupported risk tier: {risk_tier}"
            )

        return self.RISK_PRIORITY[risk_tier]

    # ============================================================
    # BUSINESS SIGNAL ESCALATION
    # ============================================================

    def _has_high_business_signal(
        self,
        business_signals: List[Dict[str, Any]],
    ) -> bool:

        for signal in business_signals:

            if signal.get("signal") == "HIGH":
                return True

        return False

    # ============================================================
    # PRIMARY REASON
    # ============================================================

    def _build_reason(
        self,
        risk_tier: str,
        projected_churn: float,
        business_signals: List[Dict[str, Any]],
        explanation: Dict[str, Any],
    ) -> str:

        risk_drivers = explanation.get(
            "risk_drivers",
            [],
        )

        if risk_drivers:

            primary_driver = risk_drivers[0]

            return (
                f"Projected churn is {projected_churn:.2f}% "
                f"with {risk_tier.lower()} risk. "
                f"The strongest modeled risk driver is "
                f"{primary_driver.get('feature', 'an identified feature')}."
            )

        if self._has_high_business_signal(
            business_signals
        ):

            return (
                f"Projected churn is {projected_churn:.2f}% "
                f"and the customer has high-priority "
                f"business signals requiring attention."
            )

        return (
            f"Projected churn is {projected_churn:.2f}% "
            f"with {risk_tier.lower()} risk."
        )

    # ============================================================
    # ACTION GENERATION
    # ============================================================

    def _generate_actions(
        self,
        priority: str,
        business_signals: List[Dict[str, Any]],
        business_decision: Dict[str, Any],
    ) -> List[str]:

        actions: List[str] = []

        # --------------------------------------------------------
        # Core workflow action
        # --------------------------------------------------------

        if priority == "URGENT":

            actions.append(
                "Route customer to immediate human retention outreach."
            )

        elif priority == "HIGH":

            actions.append(
                "Create a customer-success follow-up task."
            )

        elif priority == "MEDIUM":

            actions.append(
                "Enroll customer in a targeted engagement workflow."
            )

        else:

            actions.append(
                "Continue normal customer monitoring."
            )

        # --------------------------------------------------------
        # Business-rule recommendations
        # --------------------------------------------------------

        business_actions = business_decision.get(
            "recommended_actions",
            [],
        )

        for action in business_actions:

            if action not in actions:
                actions.append(action)

        # --------------------------------------------------------
        # High business signals
        # --------------------------------------------------------

        for signal in business_signals:

            if signal.get("signal") == "HIGH":

                recommended_action = signal.get(
                    "recommended_action"
                )

                if (
                    recommended_action
                    and recommended_action not in actions
                ):
                    actions.append(
                        recommended_action
                    )

        return actions

    # ============================================================
    # MAIN EVALUATION
    # ============================================================

    def evaluate(
        self,
        risk_tier: str,
        projected_churn: float,
        business_decision: Dict[str, Any],
        explanation: Dict[str, Any],
    ) -> Dict[str, Any]:

        # --------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------

        try:
            projected_churn = float(
                projected_churn
            )
        except (TypeError, ValueError) as exc:

            raise ValueError(
                "Projected churn must be numeric."
            ) from exc

        if not 0 <= projected_churn <= 100:

            raise ValueError(
                "Projected churn must be between 0 and 100."
            )

        if not isinstance(
            business_decision,
            dict,
        ):

            raise ValueError(
                "Business decision must be a dictionary."
            )

        if not isinstance(
            explanation,
            dict,
        ):

            raise ValueError(
                "Explanation must be a dictionary."
            )

        # --------------------------------------------------------
        # Base priority
        # --------------------------------------------------------

        priority = self._base_priority(
            risk_tier
        )

        business_signals = (
            business_decision.get(
                "business_signals",
                [],
            )
        )

        # --------------------------------------------------------
        # Build operational reason
        # --------------------------------------------------------

        reason = self._build_reason(
            risk_tier=risk_tier,
            projected_churn=projected_churn,
            business_signals=business_signals,
            explanation=explanation,
        )

        # --------------------------------------------------------
        # Generate actions
        # --------------------------------------------------------

        actions = self._generate_actions(
            priority=priority,
            business_signals=business_signals,
            business_decision=business_decision,
        )

        # --------------------------------------------------------
        # Final operational output
        # --------------------------------------------------------

        return {
            "priority": priority,

            "priority_rank": self.PRIORITY_ORDER[
                priority
            ],

            "action_window": self.ACTION_WINDOWS[
                priority
            ],

            "workflow": self.WORKFLOWS[
                priority
            ],

            "reason": reason,

            "recommended_actions": actions,

            "action_count": len(actions),
        }