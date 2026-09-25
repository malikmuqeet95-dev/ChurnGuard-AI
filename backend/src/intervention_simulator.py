from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List


class InterventionSimulator:
    """
    Runs model-based what-if simulations.

    Important:
    These simulations show how the trained model's prediction changes
    when selected customer attributes are modified.

    They should NOT be interpreted as causal treatment effects.
    """

    # Only fields that make reasonable sense as controllable / scenario
    # variables are allowed to be changed by the simulator.
    ALLOWED_INTERVENTION_FIELDS = {
        "MonthlyCharges",
        "Contract",
        "PaymentMethod",
        "PaperlessBilling",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    }

    CATEGORICAL_OPTIONS = {
        "Contract": {
            "Month-to-month",
            "One year",
            "Two year",
        },
        "PaymentMethod": {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        },
        "PaperlessBilling": {
            "Yes",
            "No",
        },
        "MultipleLines": {
            "Yes",
            "No",
            "No phone service",
        },
        "OnlineSecurity": {
            "Yes",
            "No",
            "No internet service",
        },
        "OnlineBackup": {
            "Yes",
            "No",
            "No internet service",
        },
        "DeviceProtection": {
            "Yes",
            "No",
            "No internet service",
        },
        "TechSupport": {
            "Yes",
            "No",
            "No internet service",
        },
        "StreamingTV": {
            "Yes",
            "No",
            "No internet service",
        },
        "StreamingMovies": {
            "Yes",
            "No",
            "No internet service",
        },
    }

    def __init__(
        self,
        prediction_callable: Callable[[Dict[str, Any]], Dict[str, Any]],
    ):
        """
        Parameters
        ----------
        prediction_callable:
            Function that receives a customer dictionary and returns
            the normal ChurnGuard prediction response.
        """
        self.prediction_callable = prediction_callable

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_scenario(
        self,
        baseline_customer: Dict[str, Any],
        scenario_name: str,
        changes: Dict[str, Any],
    ) -> None:

        if not isinstance(scenario_name, str) or not scenario_name.strip():
            raise ValueError(
                "Each intervention scenario must have a non-empty name."
            )

        if not isinstance(changes, dict):
            raise ValueError(
                f"Scenario '{scenario_name}' changes must be a dictionary."
            )

        if not changes:
            raise ValueError(
                f"Scenario '{scenario_name}' must contain at least one change."
            )

        for field, value in changes.items():

            if field not in self.ALLOWED_INTERVENTION_FIELDS:
                allowed = ", ".join(
                    sorted(self.ALLOWED_INTERVENTION_FIELDS)
                )

                raise ValueError(
                    f"Field '{field}' cannot be changed by the intervention "
                    f"simulator. Allowed fields: {allowed}"
                )

            if field not in baseline_customer:
                raise ValueError(
                    f"Field '{field}' does not exist in the baseline customer."
                )

            # ----------------------------------------------------
            # Numerical validation
            # ----------------------------------------------------

            if field == "MonthlyCharges":

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "MonthlyCharges intervention must be numeric."
                    ) from exc

                if numeric_value <= 0:
                    raise ValueError(
                        "MonthlyCharges intervention must be greater than 0."
                    )

            # ----------------------------------------------------
            # Categorical validation
            # ----------------------------------------------------

            if field in self.CATEGORICAL_OPTIONS:

                valid_values = self.CATEGORICAL_OPTIONS[field]

                if value not in valid_values:

                    readable_values = ", ".join(
                        sorted(valid_values)
                    )

                    raise ValueError(
                        f"Invalid value '{value}' for intervention field "
                        f"'{field}'. Valid values: {readable_values}"
                    )

        # --------------------------------------------------------
        # Reject complete no-op scenarios
        # --------------------------------------------------------

        changed_something = any(
            baseline_customer.get(field) != value
            for field, value in changes.items()
        )

        if not changed_something:
            raise ValueError(
                f"Scenario '{scenario_name}' does not change any "
                f"baseline customer value."
            )

    # ============================================================
    # PREDICTION SNAPSHOT
    # ============================================================

    def _create_snapshot(
        self,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:

        return {
            "hazard_ratio": float(prediction["hazard_ratio"]),
            "projected_churn": float(prediction["projected_churn"]),
            "projected_retention": float(
                prediction["projected_retention"]
            ),
            "risk_tier": str(prediction["risk_tier"]),
        }

    # ============================================================
    # IMPACT CLASSIFICATION
    # ============================================================

    def _classify_impact(
        self,
        churn_delta_pp: float,
    ) -> str:
        """
        churn_delta_pp:
            scenario churn % - baseline churn %

        Negative values = modeled improvement.
        Positive values = modeled deterioration.
        """

        if churn_delta_pp <= -15:
            return "MAJOR_IMPROVEMENT"

        if churn_delta_pp <= -5:
            return "MODERATE_IMPROVEMENT"

        if churn_delta_pp < -0.01:
            return "MINOR_IMPROVEMENT"

        if churn_delta_pp >= 5:
            return "ADVERSE"

        if churn_delta_pp > 0.01:
            return "SLIGHTLY_ADVERSE"

        return "NEUTRAL"

    # ============================================================
    # DELTA CALCULATION
    # ============================================================

    def _calculate_delta(
        self,
        baseline: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:

        baseline_hazard = float(baseline["hazard_ratio"])
        scenario_hazard = float(scenario["hazard_ratio"])

        baseline_churn = float(baseline["projected_churn"])
        scenario_churn = float(scenario["projected_churn"])

        baseline_retention = float(
            baseline["projected_retention"]
        )
        scenario_retention = float(
            scenario["projected_retention"]
        )

        hazard_delta = scenario_hazard - baseline_hazard

        if baseline_hazard != 0:
            hazard_delta_pct = (
                hazard_delta / baseline_hazard
            ) * 100
        else:
            hazard_delta_pct = 0.0

        churn_delta_pp = scenario_churn - baseline_churn

        retention_delta_pp = (
            scenario_retention - baseline_retention
        )

        return {
            "hazard_ratio_change": round(
                hazard_delta,
                3,
            ),
            "hazard_ratio_change_pct": round(
                hazard_delta_pct,
                2,
            ),
            "churn_change_pp": round(
                churn_delta_pp,
                2,
            ),
            "retention_change_pp": round(
                retention_delta_pp,
                2,
            ),
            "risk_tier_changed": (
                baseline["risk_tier"]
                != scenario["risk_tier"]
            ),
            "modeled_improvement": (
                churn_delta_pp < -0.01
            ),
            "impact": self._classify_impact(
                churn_delta_pp
            ),
        }

    # ============================================================
    # SINGLE SCENARIO
    # ============================================================

    def _simulate_single_scenario(
        self,
        baseline_customer: Dict[str, Any],
        baseline_prediction: Dict[str, Any],
        scenario_name: str,
        changes: Dict[str, Any],
    ) -> Dict[str, Any]:

        self._validate_scenario(
            baseline_customer=baseline_customer,
            scenario_name=scenario_name,
            changes=changes,
        )

        scenario_customer = deepcopy(
            baseline_customer
        )

        scenario_customer.update(
            changes
        )

        scenario_prediction = self.prediction_callable(
            scenario_customer
        )

        scenario_snapshot = self._create_snapshot(
            scenario_prediction
        )

        baseline_snapshot = self._create_snapshot(
            baseline_prediction
        )

        delta = self._calculate_delta(
            baseline=baseline_snapshot,
            scenario=scenario_snapshot,
        )

        return {
            "name": scenario_name.strip(),
            "changes": changes,
            "prediction": scenario_snapshot,
            "delta": delta,
        }

    # ============================================================
    # RANKING
    # ============================================================

    def _rank_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        ranked = sorted(
            scenarios,
            key=lambda item: item["delta"]["churn_change_pp"],
        )

        for index, scenario in enumerate(
            ranked,
            start=1,
        ):
            scenario["rank"] = index

        return ranked

    # ============================================================
    # MAIN SIMULATION
    # ============================================================

    def simulate(
        self,
        customer: Dict[str, Any],
        scenarios: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not isinstance(customer, dict):
            raise ValueError(
                "Baseline customer must be a dictionary."
            )

        if not isinstance(scenarios, list):
            raise ValueError(
                "Scenarios must be provided as a list."
            )

        if not scenarios:
            raise ValueError(
                "At least one intervention scenario is required."
            )

        if len(scenarios) > 10:
            raise ValueError(
                "A maximum of 10 intervention scenarios "
                "can be evaluated per request."
            )

        # --------------------------------------------------------
        # Calculate baseline once
        # --------------------------------------------------------

        baseline_prediction = self.prediction_callable(
            deepcopy(customer)
        )

        baseline_snapshot = self._create_snapshot(
            baseline_prediction
        )

        scenario_results: List[
            Dict[str, Any]
        ] = []

        for scenario in scenarios:

            if not isinstance(scenario, dict):
                raise ValueError(
                    "Each intervention scenario must be "
                    "a dictionary."
                )

            scenario_name = scenario.get(
                "name",
                "",
            )

            changes = scenario.get(
                "changes",
                {},
            )

            result = self._simulate_single_scenario(
                baseline_customer=customer,
                baseline_prediction=baseline_prediction,
                scenario_name=scenario_name,
                changes=changes,
            )

            scenario_results.append(
                result
            )

        ranked_scenarios = self._rank_scenarios(
            scenario_results
        )

        best_scenario = (
            ranked_scenarios[0]
            if ranked_scenarios
            else None
        )

        best_scenario_name = (
            best_scenario["name"]
            if best_scenario
            and best_scenario["delta"][
                "modeled_improvement"
            ]
            else None
        )

        return {
            "status": "success",
            "baseline": baseline_snapshot,
            "scenarios": ranked_scenarios,
            "scenario_count": len(
                ranked_scenarios
            ),
            "best_scenario": best_scenario_name,
            "disclaimer": (
                "What-if results are model-based counterfactual "
                "simulations. They indicate how the trained model's "
                "prediction changes when inputs are modified and "
                "should not be interpreted as causal treatment effects."
            ),
        }