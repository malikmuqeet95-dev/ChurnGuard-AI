from pydantic import BaseModel, ConfigDict, Field
from typing import Any


class CustomerPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    tenure: int = Field(
        default=3,
        ge=1,
        le=72,
        description="Current customer tenure in months.",
    )

    forecast_horizon: int = Field(
        default=6,
        ge=1,
        le=36,
        description="Number of future months to forecast.",
    )

    MonthlyCharges: float = Field(
        default=75.0,
        ge=10.0,
        le=250.0,
        description="Current monthly customer charges.",
    )

    TotalCharges: float = Field(
        default=225.0,
        ge=0.0,
        description="Total customer charges to date.",
    )

    SeniorCitizen: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    gender: str = Field(
        default="Male"
    )

    Partner: str = Field(
        default="No"
    )

    Dependents: str = Field(
        default="No"
    )

    PhoneService: str = Field(
        default="Yes"
    )

    MultipleLines: str = Field(
        default="No"
    )

    InternetService: str = Field(
        default="Fiber optic"
    )

    OnlineSecurity: str = Field(
        default="No"
    )

    OnlineBackup: str = Field(
        default="No"
    )

    DeviceProtection: str = Field(
        default="No"
    )

    TechSupport: str = Field(
        default="No"
    )

    StreamingTV: str = Field(
        default="No"
    )

    StreamingMovies: str = Field(
        default="No"
    )

    Contract: str = Field(
        default="Month-to-month"
    )

    PaperlessBilling: str = Field(
        default="Yes"
    )

    PaymentMethod: str = Field(
        default="Electronic check"
    )


# ============================================================
# EXPLAINABILITY SCHEMAS
# ============================================================


class RiskDriver(BaseModel):
    feature: str
    model_feature: str
    value: str
    coefficient: float
    contribution: float
    hazard_multiplier: float
    direction: str
    impact: str
    description: str


class Explanation(BaseModel):
    top_drivers: list[RiskDriver]
    risk_drivers: list[RiskDriver]
    protective_drivers: list[RiskDriver]
    driver_count: int


# ============================================================
# FORECAST SCHEMAS
# ============================================================


class ForecastCurvePoint(BaseModel):
    month: float
    retention: float
    churn: float


class ForecastCurve(BaseModel):
    timeline: list[float]
    survival_prob: list[float]
    churn_prob: list[float]


# ============================================================
# PREDICTION RESPONSE
# ============================================================

class BusinessSignal(BaseModel):
    category: str
    signal: str
    description: str
    recommended_action: str


class BusinessDecision(BaseModel):
    priority: str
    strategy: str
    objective: str
    lifecycle_stage: str
    primary_reason: str
    business_signals: list[BusinessSignal]
    recommended_actions: list[str]
    action_count: int

class ActionPriority(BaseModel):
    priority: str
    priority_rank: int
    action_window: str
    workflow: str
    reason: str
    recommended_actions: list[str]
    action_count: int

class RetentionIntervention(BaseModel):
    intervention_id: str
    name: str
    category: str
    reason: str
    eligibility: str
    priority: str
    channel: str
    objective: str
    recommended_action: str
    simulation_changes: dict[str, Any]
    rank: int

class RetentionInterventionROI(BaseModel):
    intervention_id: str
    estimated_cost: float
    customer_value_assumption: float
    modeled_churn_improvement_pp: float
    estimated_value_preserved: float
    estimated_net_value: float
    estimated_roi_pct: float
    economic_recommendation: str
    disclaimer: str


class EvaluatedRetentionIntervention(BaseModel):
    intervention_id: str
    name: str
    category: str
    reason: str
    eligibility: str
    priority: str
    channel: str
    objective: str
    recommended_action: str
    simulation_changes: dict[str, Any]
    rank: int
    roi: RetentionInterventionROI | None = None
    decision_rank: int


class SelectedRetentionIntervention(BaseModel):
    intervention_id: str
    name: str
    category: str
    priority: str
    channel: str
    recommended_action: str
    simulation_changes: dict[str, Any]
    roi: RetentionInterventionROI


class RetentionDecision(BaseModel):
    decision_status: str
    risk_tier: str
    projected_churn: float
    operational_priority: str
    recommendation: str
    reason: str
    selected_intervention: (
        SelectedRetentionIntervention | None
    )
    evaluated_interventions: list[
        EvaluatedRetentionIntervention
    ]
    evaluated_count: int
    decision_disclaimer: str


class InterventionScenarioInput(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    changes: dict[str, Any]


class WhatIfRequest(BaseModel):
    customer: CustomerPayload

    scenarios: list[
        InterventionScenarioInput
    ] = Field(
        ...,
        min_length=1,
        max_length=10,
    )


class PredictionSnapshot(BaseModel):
    hazard_ratio: float
    projected_churn: float
    projected_retention: float
    risk_tier: str


class InterventionDelta(BaseModel):
    hazard_ratio_change: float
    hazard_ratio_change_pct: float

    churn_change_pp: float
    retention_change_pp: float

    risk_tier_changed: bool
    modeled_improvement: bool

    impact: str


class InterventionResult(BaseModel):
    name: str

    changes: dict[str, Any]

    prediction: PredictionSnapshot

    delta: InterventionDelta

    rank: int


class WhatIfResponse(BaseModel):
    status: str

    baseline: PredictionSnapshot

    scenarios: list[
        InterventionResult
    ]

    scenario_count: int

    best_scenario: str | None

    disclaimer: str


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str

    status: str
    current_tenure: int
    forecast_horizon: int
    target_month: int
    hazard_ratio: float
    projected_churn: float
    projected_retention: float
    risk_tier: str
    recommended_action: str

    forecast_curve: dict[str, list[float]]

    explanation: Explanation

    business_decision: BusinessDecision

    action_priority: ActionPriority

    retention_interventions: list[
        RetentionIntervention
    ]

    retention_decision: RetentionDecision
    