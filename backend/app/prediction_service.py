from __future__ import annotations

from typing import Any
from unittest import result

from backend.app.config import settings
from backend.app.performance import PerformanceTimer
from backend.app.model_metadata import ModelMetadataService
from backend.app.logging_config import get_logger
from backend.src.action_priority import ActionPriorityEngine
from backend.src.business_rules import BusinessRiskEngine
from backend.src.explainability import RiskExplainer
from backend.src.intervention_library import (
    RetentionInterventionLibrary,
)
from backend.src.intervention_roi import (
    InterventionROIEngine,
)
from backend.src.intervention_simulator import (
    InterventionSimulator,
)
from backend.src.retention_decision import (
    RetentionDecisionEngine,
)
from backend.src.survival_predictor import (
    SurvivalPredictor,
)

logger = get_logger(__name__)

class PredictionService:
    """
    Orchestrates the complete customer retention
    decision-intelligence pipeline.

    Pipeline:

        Survival Prediction
            ↓
        Explainability
            ↓
        Business Decision
            ↓
        Action Priority
            ↓
        Intervention Library
            ↓
        What-if Simulation
            ↓
        Economic Evaluation
            ↓
        Final Retention Decision

    The underlying ML model remains unchanged.

    Important:
    What-if simulations and ROI estimates are decision-support
    tools. They are not causal treatment-effect estimates.
    """

    def __init__(self):
        # ---------------------------------------------------------
        # CORE ML COMPONENT
        # ---------------------------------------------------------
        self.predictor = SurvivalPredictor()

        # ---------------------------------------------------------
        # EXPLAINABILITY
        # ---------------------------------------------------------
        self.explainer = RiskExplainer(
            self.predictor.model
        )

        # ---------------------------------------------------------
        # BUSINESS RULES
        # ---------------------------------------------------------
        self.business_engine = BusinessRiskEngine()

        # ---------------------------------------------------------
        # ACTION PRIORITY
        # ---------------------------------------------------------
        self.action_priority_engine = (
            ActionPriorityEngine()
        )

        # ---------------------------------------------------------
        # RETENTION INTERVENTION LIBRARY
        # ---------------------------------------------------------
        self.intervention_library = (
            RetentionInterventionLibrary()
        )

        # ---------------------------------------------------------
        # INTERVENTION ROI
        # ---------------------------------------------------------
        self.intervention_roi_engine = (
            InterventionROIEngine()
        )

        self.model_metadata_service = ModelMetadataService()

        # ---------------------------------------------------------
        # FINAL RETENTION DECISION
        # ---------------------------------------------------------
        self.retention_decision_engine = (
            RetentionDecisionEngine()
        )

        # ---------------------------------------------------------
        # WHAT-IF SIMULATOR
        #
        # IMPORTANT:
        # Do NOT use self.predict here.
        #
        # self.predict now contains the simulation layer itself.
        # Using self.predict would create infinite recursion.
        #
        # Instead we use the lower-level simulation prediction
        # method which only executes the underlying ML prediction.
        # ---------------------------------------------------------
        self.intervention_simulator = (
            InterventionSimulator(
                prediction_callable=(
                    self._predict_for_simulation
                )
            )
        )

    # =========================================================
    # LOW-LEVEL PREDICTION FOR WHAT-IF SIMULATIONS
    # =========================================================

    def _predict_for_simulation(
        self,
        customer_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Produce the minimal prediction snapshot required by
        InterventionSimulator.

        This method intentionally does NOT execute:

            - explainability
            - business rules
            - intervention library
            - ROI
            - retention decision

        This prevents recursive calls when simulations are
        executed inside the main prediction pipeline.
        """

        result = self.predictor.predict_risk_profile(
            customer_data
        )

        return {
            "hazard_ratio": float(
                result["hazard_ratio_multiplier"]
            ),
            "projected_churn": float(
                result["projected_churn_pct"]
            ),
            "projected_retention": float(
                result["projected_retention_pct"]
            ),
            "risk_tier": str(
                result["risk_tier"]
            ),
        }

    # =========================================================
    # BASE MODEL + EXPLANATION + BUSINESS DECISION
    # =========================================================

    def _build_base_prediction(
        self,
        customer_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute the prediction layers that do not require
        intervention simulation.

        Returns the normal prediction plus:

            - explanation
            - business decision
            - action priority
            - retention intervention candidates
        """
        model_metadata = self.model_metadata_service.get_metadata()

        # -----------------------------------------------------
        # 1. SURVIVAL MODEL
        # -----------------------------------------------------
        result = self.predictor.predict_risk_profile(
            customer_data
        )

        # -----------------------------------------------------
        # 2. EXPLAINABILITY
        # -----------------------------------------------------
        processed_features = (
            self.predictor.preprocess_input(
                customer_data
            )
        )

        model_features = list(
            self.predictor.model.params_.index
        )

        processed_features = processed_features[
            model_features
        ]

        explanation = self.explainer.explain(
            processed_features=processed_features,
            customer_input=customer_data,
            top_n=5,
        )

        # -----------------------------------------------------
        # 3. BUSINESS DECISION
        # -----------------------------------------------------
        business_decision = (
            self.business_engine.evaluate(
                customer=customer_data,
                risk_tier=result["risk_tier"],
                projected_churn=float(
                    result["projected_churn_pct"]
                ),
                explanation=explanation,
            )
        )

        # -----------------------------------------------------
        # 4. ACTION PRIORITY
        # -----------------------------------------------------
        action_priority = (
            self.action_priority_engine.evaluate(
                risk_tier=result["risk_tier"],
                projected_churn=float(
                    result["projected_churn_pct"]
                ),
                business_decision=business_decision,
                explanation=explanation,
            )
        )

        # -----------------------------------------------------
        # 5. RETENTION INTERVENTION CANDIDATES
        # -----------------------------------------------------
        retention_interventions = (
            self.intervention_library.get_candidates(
                customer=customer_data,
                risk_tier=str(
                    result["risk_tier"]
                ),
                projected_churn=float(
                    result["projected_churn_pct"]
                ),
                business_decision=business_decision,
                explanation=explanation,
            )
        )

        # -----------------------------------------------------
        # 6. SURVIVAL CURVE
        # -----------------------------------------------------
        curve = result.get(
            "survival_curve",
            [],
        )

        timeline = []
        survival_prob = []
        churn_prob = []

        for point in curve:
            timeline.append(
                float(point["month"])
            )

            survival_prob.append(
                round(
                    float(
                        point["retention"]
                    ),
                    4,
                )
            )

            churn_prob.append(
                round(
                    float(
                        point["churn"]
                    ),
                    4,
                )
            )

        return {
            "status": "success",
            "model_version": model_metadata["model_version"],
            "model_name": model_metadata["model_name"],

            "current_tenure": int(
                result[
                    "current_tenure_months"
                ]
            ),
            "forecast_horizon": int(
                result[
                    "forecast_horizon_months"
                ]
            ),
            "target_month": int(
                result["target_month"]
            ),
            "hazard_ratio": float(
                result[
                    "hazard_ratio_multiplier"
                ]
            ),
            "projected_churn": float(
                result[
                    "projected_churn_pct"
                ]
            ),
            "projected_retention": float(
                result[
                    "projected_retention_pct"
                ]
            ),
            "risk_tier": str(
                result["risk_tier"]
            ),
            "recommended_action": str(
                result["recommended_action"]
            ),
            "forecast_curve": {
                "timeline": timeline,
                "survival_prob": survival_prob,
                "churn_prob": churn_prob,
            },
            "explanation": explanation,
            "business_decision": business_decision,
            "action_priority": action_priority,
            "retention_interventions": (
                retention_interventions
            ),
        }

    # =========================================================
    # INTERVENTION SIMULATION
    # =========================================================

    def _simulate_retention_interventions(
        self,
        customer_data: dict[str, Any],
        retention_interventions: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:
        """
        Run model-based what-if simulations for all
        intervention candidates that have explicit
        simulation mappings.

        Some interventions intentionally have no model
        simulation mapping, for example:

            - onboarding support
            - loyalty recognition

        Those remain operational recommendations but are
        not assigned artificial model effects.
        """

        scenarios = []

        for intervention in retention_interventions:
            changes = intervention.get(
                "simulation_changes",
                {},
            )

            # -------------------------------------------------
            # No model-change mapping.
            #
            # Do not fabricate a simulation.
            # -------------------------------------------------
            if not changes:
                continue

            scenarios.append(
                {
                    "name": intervention[
                        "intervention_id"
                    ],
                    "changes": changes,
                }
            )

        if not scenarios:
            return []

        simulation_result = (
            self.intervention_simulator.simulate(
                customer=customer_data,
                scenarios=scenarios,
            )
        )

        completed_simulations = (
            simulation_result.get(
                "scenarios",
                [],
            )
        )

        # -----------------------------------------------------
        # Add intervention_id to each simulation.
        #
        # The simulator uses scenario name as the intervention
        # identifier because the intervention library gives
        # every intervention a stable ID.
        # -----------------------------------------------------
        enriched = []

        for simulation in completed_simulations:
            simulation_copy = dict(
                simulation
            )

            simulation_copy[
                "intervention_id"
            ] = simulation_copy.get(
                "name"
            )

            enriched.append(
                simulation_copy
            )

        return enriched

    # =========================================================
    # ROI ENRICHMENT
    # =========================================================

    def _attach_roi_to_interventions(
        self,
        retention_interventions: list[
            dict[str, Any]
        ],
        simulations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Attach estimated economic results to the corresponding
        intervention candidates.

        Interventions without simulations remain in the list,
        but do not receive fabricated ROI values.
        """

        if not simulations:
            return [
                dict(intervention)
                for intervention
                in retention_interventions
            ]

        # -----------------------------------------------------
        # Calculate ROI for completed simulations.
        # -----------------------------------------------------
        roi_ranked = (
            self.intervention_roi_engine.rank(
                simulations=simulations,
            )
        )

        roi_by_id = {}

        for simulation in roi_ranked:
            intervention_id = simulation.get(
                "intervention_id"
            )

            if intervention_id:
                roi_by_id[
                    intervention_id
                ] = simulation.get(
                    "roi",
                    {},
                )

        # -----------------------------------------------------
        # Attach ROI to library candidates.
        # -----------------------------------------------------
        enriched = []

        for intervention in retention_interventions:
            intervention_copy = dict(
                intervention
            )

            intervention_id = (
                intervention_copy.get(
                    "intervention_id"
                )
            )

            if intervention_id in roi_by_id:
                intervention_copy[
                    "roi"
                ] = roi_by_id[
                    intervention_id
                ]

            enriched.append(
                intervention_copy
            )

        return enriched

    # =========================================================
    # FINAL RETENTION DECISION
    # =========================================================

    def _build_retention_decision(
        self,
        base_prediction: dict[str, Any],
        interventions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Generate the final operational retention decision.
        """

        return (
            self.retention_decision_engine.decide(
                risk_tier=base_prediction[
                    "risk_tier"
                ],
                projected_churn=float(
                    base_prediction[
                        "projected_churn"
                    ]
                ),
                action_priority=(
                    base_prediction[
                        "action_priority"
                    ]
                ),
                interventions=interventions,
            )
        )

    # =========================================================
    # PUBLIC PREDICTION API
    # =========================================================

    def predict(self, customer_data: dict[str, Any]) -> dict[str, Any]:

        timer = PerformanceTimer()
        logger.debug("Starting retention prediction pipeline")

        # -----------------------------------------------------
        # STEP 1
        # Core prediction + explanation + business logic
        # -----------------------------------------------------

        with timer.measure("base_prediction"):
            base_prediction = self._build_base_prediction(customer_data)

        logger.debug(
            "Base survival prediction completed | risk_tier=%s | churn=%.2f",
            base_prediction["risk_tier"],
            base_prediction["projected_churn"],
        )

        # -----------------------------------------------------
        # STEP 2
        # Model-based what-if simulations
        # -----------------------------------------------------
        with timer.measure("intervention_simulation"):
            simulations = self._simulate_retention_interventions(
                customer_data=customer_data,
                retention_interventions=base_prediction["retention_interventions"],
            )

        logger.debug(
            "Intervention simulation completed | scenarios=%d",
            len(simulations),
        )

        # -----------------------------------------------------
        # STEP 3
        # Attach estimated ROI
        # -----------------------------------------------------
        with timer.measure("roi"):
            enriched_interventions = self._attach_roi_to_interventions(
                retention_interventions=base_prediction["retention_interventions"],
                simulations=simulations,
            )   

        # -----------------------------------------------------
        # STEP 4
        # Final decision
        # -----------------------------------------------------
        with timer.measure("retention_decision"):
            retention_decision = self._build_retention_decision(
                base_prediction=base_prediction,
                interventions=enriched_interventions,
            )

        logger.info(
            "Retention decision generated | risk_tier=%s | priority=%s | recommendation=%s",
            base_prediction["risk_tier"],
            retention_decision["operational_priority"],
            retention_decision["recommendation"],
        )

        # -----------------------------------------------------
        # STEP 5
        # Final response
        # -----------------------------------------------------

        response = dict(base_prediction)
        response["retention_interventions"] = enriched_interventions
        response["retention_decision"] = retention_decision

        performance_timings = timer.result()

        logger.debug(
            "Prediction performance | timings_ms=%s | total_measured_ms=%.3f",
            performance_timings,
            timer.total_ms(),
        )

        if settings.is_development:
            response["performance"] = {
                "timings_ms": performance_timings,
                "total_measured_ms": timer.total_ms(),
            }
        return response
    # =========================================================
    # PUBLIC WHAT-IF API
    # =========================================================

    def simulate_interventions(
        self,
        customer_data: dict[str, Any],
        scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Preserve the existing explicit what-if endpoint.

        This continues to provide raw model-based scenario
        comparison independently from the automatic retention
        decision pipeline.
        """

        return self.intervention_simulator.simulate(
            customer=customer_data,
            scenarios=scenarios,
        )