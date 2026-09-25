from __future__ import annotations

from typing import Any, Dict, List


class BusinessRiskEngine:
    """
    Convert model outputs and customer attributes into
    business-oriented retention decisions.

    This layer does NOT change the ML prediction.

    It translates model results into operational actions.
    """

    # ========================================================
    # RISK PRIORITY
    # ========================================================

    RISK_PRIORITY = {
        "CRITICAL": "URGENT",
        "MODERATE": "HIGH",
        "HEALTHY": "NORMAL",
    }

    # ========================================================
    # CUSTOMER LIFECYCLE STAGES
    # ========================================================

    def _get_lifecycle_stage(
        self,
        tenure: int,
    ) -> str:
        """
        Determine the customer's lifecycle stage.
        """

        if tenure <= 6:
            return "EARLY_LIFE"

        if tenure <= 12:
            return "DEVELOPING"

        if tenure <= 24:
            return "ESTABLISHED"

        if tenure <= 48:
            return "MATURE"

        return "LONG_TERM"

    # ========================================================
    # CONTRACT RISK
    # ========================================================

    def _get_contract_signal(
        self,
        contract: str,
    ) -> Dict[str, Any]:
        """
        Identify contract-related retention opportunities.
        """

        if contract == "Month-to-month":
            return {
                "signal": "HIGH",
                "category": "CONTRACT",
                "description": (
                    "Customer is on a Month-to-month contract."
                ),
                "recommended_action": (
                    "Offer a targeted incentive for moving "
                    "to an annual contract."
                ),
            }

        if contract == "One year":
            return {
                "signal": "MEDIUM",
                "category": "CONTRACT",
                "description": (
                    "Customer has an annual commitment."
                ),
                "recommended_action": (
                    "Reinforce contract value and prepare "
                    "renewal engagement before expiration."
                ),
            }

        return {
            "signal": "LOW",
            "category": "CONTRACT",
            "description": (
                "Customer has a long-term contract."
            ),
            "recommended_action": (
                "Focus on loyalty and expansion opportunities."
            ),
        }

    # ========================================================
    # PAYMENT RISK
    # ========================================================

    def _get_payment_signal(
        self,
        payment_method: str,
    ) -> Dict[str, Any]:
        """
        Identify payment-method-related intervention
        opportunities.

        These are operational signals, not claims of
        causality.
        """

        if payment_method == "Electronic check":
            return {
                "signal": "HIGH",
                "category": "PAYMENT",
                "description": (
                    "Customer uses electronic check "
                    "payment."
                ),
                "recommended_action": (
                    "Consider promoting automatic payment "
                    "options and reducing payment friction."
                ),
            }

        return {
            "signal": "LOW",
            "category": "PAYMENT",
            "description": (
                "Customer uses an automated or mailed "
                "payment method."
            ),
            "recommended_action": (
                "Maintain payment convenience."
            ),
        }

    # ========================================================
    # SERVICE ENGAGEMENT SIGNALS
    # ========================================================

    def _get_service_signals(
        self,
        customer: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Identify service engagement opportunities.
        """

        signals = []

        if customer.get("TechSupport") == "No":
            signals.append(
                {
                    "category": "SERVICE",
                    "signal": "MEDIUM",
                    "description": (
                        "Technical support is not currently "
                        "enabled."
                    ),
                    "recommended_action": (
                        "Consider offering technical support "
                        "as part of a retention package."
                    ),
                }
            )

        if customer.get("OnlineSecurity") == "No":
            signals.append(
                {
                    "category": "SECURITY",
                    "signal": "LOW",
                    "description": (
                        "Online security is not currently "
                        "enabled."
                    ),
                    "recommended_action": (
                        "Consider a security-service "
                        "bundle if relevant to the customer."
                    ),
                }
            )

        if customer.get("OnlineBackup") == "No":
            signals.append(
                {
                    "category": "SERVICE",
                    "signal": "LOW",
                    "description": (
                        "Online backup is not currently "
                        "enabled."
                    ),
                    "recommended_action": (
                        "Consider a bundled service offer "
                        "where appropriate."
                    ),
                }
            )

        return signals

    # ========================================================
    # RETENTION STRATEGY
    # ========================================================

    def _get_retention_strategy(
        self,
        risk_tier: str,
        projected_churn: float,
        contract: str,
    ) -> Dict[str, Any]:
        """
        Select the overall retention strategy.
        """

        if risk_tier == "CRITICAL":

            if contract == "Month-to-month":
                strategy = (
                    "URGENT_CONTRACT_RETENTION"
                )

            else:
                strategy = (
                    "URGENT_CUSTOMER_SUCCESS"
                )

            return {
                "strategy": strategy,
                "priority": "URGENT",
                "objective": (
                    "Reduce near-term churn exposure "
                    "through targeted intervention."
                ),
            }

        if risk_tier == "MODERATE":

            return {
                "strategy": (
                    "PROACTIVE_RETENTION"
                ),
                "priority": "HIGH",
                "objective": (
                    "Address emerging churn signals "
                    "before the customer becomes critical."
                ),
            }

        return {
            "strategy": (
                "LOYALTY_AND_EXPANSION"
            ),
            "priority": "NORMAL",
            "objective": (
                "Maintain retention while identifying "
                "loyalty and expansion opportunities."
            ),
        }

    # ========================================================
    # MAIN DECISION ENGINE
    # ========================================================

    def evaluate(
        self,
        customer: Dict[str, Any],
        risk_tier: str,
        projected_churn: float,
        explanation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a business decision profile.
        """

        tenure = int(
            customer.get(
                "tenure",
                1,
            )
        )

        contract = str(
            customer.get(
                "Contract",
                "Month-to-month",
            )
        )

        payment_method = str(
            customer.get(
                "PaymentMethod",
                "",
            )
        )

        lifecycle_stage = (
            self._get_lifecycle_stage(
                tenure
            )
        )

        contract_signal = (
            self._get_contract_signal(
                contract
            )
        )

        payment_signal = (
            self._get_payment_signal(
                payment_method
            )
        )

        service_signals = (
            self._get_service_signals(
                customer
            )
        )

        strategy = (
            self._get_retention_strategy(
                risk_tier,
                projected_churn,
                contract,
            )
        )

        # ----------------------------------------------------
        # Assemble action list
        # ----------------------------------------------------

        actions: List[str] = []

        if risk_tier == "CRITICAL":

            actions.append(
                "Prioritize customer for immediate retention outreach."
            )

            if contract == "Month-to-month":

                actions.append(
                    "Present a targeted annual-contract incentive."
                )

            actions.append(
                "Review customer service and value concerns."
            )

            actions.append(
                "Consider a personalized retention offer."
            )

        elif risk_tier == "MODERATE":

            actions.append(
                "Trigger proactive customer engagement."
            )

            if contract == "Month-to-month":

                actions.append(
                    "Test an annual-contract conversion offer."
                )

            actions.append(
                "Promote relevant service features "
                "that improve customer value."
            )

        else:

            actions.append(
                "Maintain normal customer engagement."
            )

            actions.append(
                "Consider loyalty rewards or expansion offers."
            )

        # ----------------------------------------------------
        # Add relevant service action
        # ----------------------------------------------------

        if service_signals:

            actions.append(
                service_signals[0][
                    "recommended_action"
                ]
            )

        # ----------------------------------------------------
        # Primary reason
        # ----------------------------------------------------

        risk_drivers = explanation.get(
            "risk_drivers",
            [],
        )

        if risk_drivers:

            primary_driver = risk_drivers[0]

            primary_reason = (
                f"{primary_driver['feature']} "
                f"is the strongest modeled risk driver "
                f"for this customer."
            )

        else:

            primary_reason = (
                "No strong individual risk driver was "
                "identified from the explainability layer."
            )

        # ----------------------------------------------------
        # Business signals
        # ----------------------------------------------------

        business_signals = [
            contract_signal,
            payment_signal,
        ]

        business_signals.extend(
            service_signals
        )

        return {
            "priority": strategy[
                "priority"
            ],

            "strategy": strategy[
                "strategy"
            ],

            "objective": strategy[
                "objective"
            ],

            "lifecycle_stage": (
                lifecycle_stage
            ),

            "primary_reason": (
                primary_reason
            ),

            "business_signals": (
                business_signals
            ),

            "recommended_actions": (
                actions
            ),

            "action_count": len(
                actions
            ),
        }
