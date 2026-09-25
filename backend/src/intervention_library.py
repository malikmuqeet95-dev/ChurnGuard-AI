from __future__ import annotations

from typing import Any, Dict, List


class RetentionInterventionLibrary:
    """
    Converts customer context and model outputs into structured
    candidate retention interventions.

    Important:
    - This library does NOT modify the model prediction.
    - It does NOT claim that an intervention will causally reduce churn.
    - It identifies operational actions that may be considered.
    - What-if impact should be evaluated separately by the
      InterventionSimulator.
    """

    PRIORITY_ORDER = {
        "URGENT": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4,
    }

    INTERVENTION_CATALOG = {
        "CONTRACT_CONVERSION": {
            "name": "Annual Contract Conversion",
            "category": "CONTRACT",
            "channel": "CUSTOMER_SUCCESS",
            "objective": "Increase contractual commitment",
            "recommended_action": (
                "Present a targeted annual-contract offer "
                "with clear value and pricing."
            ),
            "simulation_changes": {
                "Contract": "One year",
            },
        },
        "PAYMENT_AUTOPAY": {
            "name": "Automatic Payment Migration",
            "category": "PAYMENT",
            "channel": "BILLING",
            "objective": "Reduce payment friction",
            "recommended_action": (
                "Offer automatic bank transfer or credit-card "
                "payment options."
            ),
            "simulation_changes": {
                "PaymentMethod": "Bank transfer (automatic)",
            },
        },
        "TECH_SUPPORT_BUNDLE": {
            "name": "Technical Support Offer",
            "category": "SERVICE",
            "channel": "CUSTOMER_SUCCESS",
            "objective": "Improve service support coverage",
            "recommended_action": (
                "Offer technical support or include it in a "
                "targeted retention bundle."
            ),
            "simulation_changes": {
                "TechSupport": "Yes",
            },
        },
        "ONLINE_SECURITY_BUNDLE": {
            "name": "Online Security Offer",
            "category": "SECURITY",
            "channel": "CUSTOMER_SUCCESS",
            "objective": "Increase perceived service value",
            "recommended_action": (
                "Offer online security as part of a relevant "
                "service bundle."
            ),
            "simulation_changes": {
                "OnlineSecurity": "Yes",
            },
        },
        "ONBOARDING_SUPPORT": {
            "name": "Early-Life Customer Onboarding",
            "category": "ONBOARDING",
            "channel": "CUSTOMER_SUCCESS",
            "objective": "Strengthen early customer engagement",
            "recommended_action": (
                "Trigger proactive onboarding assistance, "
                "product education, and support outreach."
            ),
            "simulation_changes": {},
        },
        "LOYALTY_RECOGNITION": {
            "name": "Loyalty Recognition",
            "category": "LOYALTY",
            "channel": "MARKETING",
            "objective": "Maintain long-term customer engagement",
            "recommended_action": (
                "Consider loyalty recognition, rewards, or "
                "relevant expansion opportunities."
            ),
            "simulation_changes": {},
        },
    }

    def _build_candidate(
        self,
        intervention_id: str,
        reason: str,
        priority: str,
    ) -> Dict[str, Any]:
        catalog_item = self.INTERVENTION_CATALOG[intervention_id]

        return {
            "intervention_id": intervention_id,
            "name": catalog_item["name"],
            "category": catalog_item["category"],
            "reason": reason,
            "eligibility": "ELIGIBLE",
            "priority": priority,
            "channel": catalog_item["channel"],
            "objective": catalog_item["objective"],
            "recommended_action": catalog_item["recommended_action"],
            "simulation_changes": dict(
                catalog_item["simulation_changes"]
            ),
        }

    def _add_contract_intervention(
        self,
        customer: Dict[str, Any],
        risk_tier: str,
        candidates: List[Dict[str, Any]],
    ) -> None:
        contract = str(
            customer.get("Contract", "Month-to-month")
        )

        if contract != "Month-to-month":
            return

        if risk_tier == "CRITICAL":
            priority = "URGENT"
        elif risk_tier == "MODERATE":
            priority = "HIGH"
        else:
            priority = "MEDIUM"

        candidates.append(
            self._build_candidate(
                intervention_id="CONTRACT_CONVERSION",
                reason=(
                    "Customer is currently on a month-to-month "
                    "contract, creating an opportunity to evaluate "
                    "a longer-term commitment offer."
                ),
                priority=priority,
            )
        )

    def _add_payment_intervention(
        self,
        customer: Dict[str, Any],
        risk_tier: str,
        candidates: List[Dict[str, Any]],
    ) -> None:
        payment_method = str(
            customer.get("PaymentMethod", "")
        )

        if payment_method != "Electronic check":
            return

        if risk_tier == "CRITICAL":
            priority = "HIGH"
        elif risk_tier == "MODERATE":
            priority = "MEDIUM"
        else:
            priority = "LOW"

        candidates.append(
            self._build_candidate(
                intervention_id="PAYMENT_AUTOPAY",
                reason=(
                    "Customer uses electronic check payment. "
                    "A payment-method migration may be considered "
                    "to reduce billing friction."
                ),
                priority=priority,
            )
        )

    def _add_service_interventions(
        self,
        customer: Dict[str, Any],
        risk_tier: str,
        candidates: List[Dict[str, Any]],
    ) -> None:
        internet_service = str(
            customer.get("InternetService", "")
        )

        if internet_service == "No":
            return

        if customer.get("TechSupport") == "No":
            priority = (
                "HIGH"
                if risk_tier == "CRITICAL"
                else "MEDIUM"
            )

            candidates.append(
                self._build_candidate(
                    intervention_id="TECH_SUPPORT_BUNDLE",
                    reason=(
                        "Technical support is not currently enabled. "
                        "A support offer may address service coverage "
                        "and customer-support needs."
                    ),
                    priority=priority,
                )
            )

        if customer.get("OnlineSecurity") == "No":
            candidates.append(
                self._build_candidate(
                    intervention_id="ONLINE_SECURITY_BUNDLE",
                    reason=(
                        "Online security is not currently enabled. "
                        "A security-service offer may increase the "
                        "customer's service coverage."
                    ),
                    priority="MEDIUM",
                )
            )

    def _add_onboarding_intervention(
        self,
        customer: Dict[str, Any],
        risk_tier: str,
        candidates: List[Dict[str, Any]],
    ) -> None:
        try:
            tenure = int(customer.get("tenure", 1))
        except (TypeError, ValueError):
            return

        if tenure > 6:
            return

        if risk_tier == "CRITICAL":
            priority = "URGENT"
        elif risk_tier == "MODERATE":
            priority = "HIGH"
        else:
            priority = "MEDIUM"

        candidates.append(
            self._build_candidate(
                intervention_id="ONBOARDING_SUPPORT",
                reason=(
                    "Customer is in the early lifecycle stage. "
                    "Proactive onboarding and education can be "
                    "considered to strengthen engagement."
                ),
                priority=priority,
            )
        )

    def _add_loyalty_intervention(
        self,
        customer: Dict[str, Any],
        risk_tier: str,
        candidates: List[Dict[str, Any]],
    ) -> None:
        try:
            tenure = int(customer.get("tenure", 1))
        except (TypeError, ValueError):
            return

        if risk_tier != "HEALTHY":
            return

        if tenure < 24:
            return

        candidates.append(
            self._build_candidate(
                intervention_id="LOYALTY_RECOGNITION",
                reason=(
                    "Customer has a relatively long tenure and "
                    "currently falls into the healthy risk tier. "
                    "Loyalty recognition may help maintain engagement."
                ),
                priority="LOW",
            )
        )

    def _deduplicate(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        unique_candidates = []

        for candidate in candidates:
            intervention_id = candidate["intervention_id"]

            if intervention_id in seen:
                continue

            seen.add(intervention_id)
            unique_candidates.append(candidate)

        return unique_candidates

    def _rank(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            candidates,
            key=lambda item: (
                self.PRIORITY_ORDER.get(
                    item["priority"],
                    99,
                ),
                item["name"],
            ),
        )

        for index, candidate in enumerate(
            ranked,
            start=1,
        ):
            candidate["rank"] = index

        return ranked

    def get_candidates(
        self,
        customer: Dict[str, Any],
        risk_tier: str,
        projected_churn: float,
        business_decision: Dict[str, Any] | None = None,
        explanation: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Return operational retention intervention candidates.

        projected_churn, business_decision, and explanation are
        accepted so the library can evolve toward richer decisioning,
        but intervention eligibility is intentionally based on
        explicit business context rather than pretending that
        interventions are causal effects.
        """

        if not isinstance(customer, dict):
            raise ValueError(
                "Customer must be provided as a dictionary."
            )

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

        if business_decision is not None and not isinstance(
            business_decision,
            dict,
        ):
            raise ValueError(
                "Business decision must be a dictionary."
            )

        if explanation is not None and not isinstance(
            explanation,
            dict,
        ):
            raise ValueError(
                "Explanation must be a dictionary."
            )

        candidates: List[Dict[str, Any]] = []

        self._add_contract_intervention(
            customer,
            risk_tier,
            candidates,
        )

        self._add_payment_intervention(
            customer,
            risk_tier,
            candidates,
        )

        self._add_service_interventions(
            customer,
            risk_tier,
            candidates,
        )

        self._add_onboarding_intervention(
            customer,
            risk_tier,
            candidates,
        )

        self._add_loyalty_intervention(
            customer,
            risk_tier,
            candidates,
        )

        candidates = self._deduplicate(candidates)

        return self._rank(candidates)