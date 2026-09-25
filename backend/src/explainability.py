from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class RiskExplainer:
    """
    Explain Cox Proportional Hazards predictions using the
    trained model coefficients and the customer's processed
    feature values.

    The explainer does not train a new model and does not
    modify the prediction model.

    It only interprets the existing Cox model.
    """

    # --------------------------------------------------------
    # BUSINESS-FRIENDLY FEATURE NAMES
    # --------------------------------------------------------

    FEATURE_LABELS = {
        "SeniorCitizen": "Senior Citizen",
        "MonthlyCharges_scaled": "Monthly Charges",
        "TotalCharges_scaled": "Total Charges",
        "gender": "Gender",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "Contract": "Contract",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method",
    }

    # --------------------------------------------------------
    # SPECIAL LABEL HANDLING
    # --------------------------------------------------------

    VALUE_LABELS = {
        "SeniorCitizen_0": "Non-senior customer",
        "SeniorCitizen_1": "Senior customer",
    }

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def __init__(self, model):
        """
        Parameters
        ----------
        model:
            Fitted lifelines CoxPHFitter model.
        """

        if model is None:
            raise ValueError(
                "A fitted Cox model is required."
            )

        if not hasattr(model, "params_"):
            raise ValueError(
                "The supplied model does not contain Cox coefficients."
            )

        self.model = model

        self.coefficients = (
            model.params_
            .astype(float)
            .copy()
        )

    # ========================================================
    # FEATURE NAME PARSING
    # ========================================================

    def _get_base_feature_name(
        self,
        feature_name: str,
    ) -> str:
        """
        Convert an encoded feature name into its original
        business feature name.

        Examples
        --------
        Contract_One year
            -> Contract

        InternetService_Fiber optic
            -> Internet Service

        MonthlyCharges_scaled
            -> Monthly Charges
        """

        if feature_name in self.FEATURE_LABELS:
            return self.FEATURE_LABELS[feature_name]

        if feature_name.endswith("_scaled"):

            original_name = (
                feature_name[:-7]
            )

            return self.FEATURE_LABELS.get(
                feature_name,
                self._humanize_name(
                    original_name
                ),
            )

        # ----------------------------------------------------
        # Categorical dummy variable
        # ----------------------------------------------------

        known_prefixes = sorted(
            [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
            ],
            key=len,
            reverse=True,
        )

        for prefix in known_prefixes:

            prefix_with_separator = (
                prefix + "_"
            )

            if feature_name.startswith(
                prefix_with_separator
            ):

                return self.FEATURE_LABELS.get(
                    prefix,
                    self._humanize_name(prefix),
                )

        return self._humanize_name(
            feature_name
        )

    # ========================================================
    # HUMAN READABLE NAME
    # ========================================================

    @staticmethod
    def _humanize_name(
        value: str,
    ) -> str:
        """
        Convert technical feature names into readable labels.
        """

        value = value.replace(
            "_scaled",
            "",
        )

        value = value.replace(
            "_",
            " ",
        )

        return value.strip().title()

    # ========================================================
    # VALUE EXTRACTION
    # ========================================================

    def _get_feature_value_description(
        self,
        feature_name: str,
        customer_input: Dict[str, Any],
    ) -> str:
        """
        Determine the customer-facing value represented by
        a model feature.
        """

        if feature_name in customer_input:

            value = customer_input[
                feature_name
            ]

            return str(value)

        # ----------------------------------------------------
        # Scaled numerical feature
        # ----------------------------------------------------

        if feature_name.endswith("_scaled"):

            original_name = (
                feature_name[:-7]
            )

            if original_name in customer_input:

                return str(
                    customer_input[
                        original_name
                    ]
                )

        # ----------------------------------------------------
        # Categorical dummy variable
        # ----------------------------------------------------

        categorical_columns = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]

        for column in categorical_columns:

            prefix = column + "_"

            if feature_name.startswith(
                prefix
            ):

                category = feature_name[
                    len(prefix):
                ]

                actual_value = customer_input.get(
                    column
                )

                if actual_value is None:
                    return category

                if str(actual_value) == category:

                    return category

                return (
                    f"{actual_value} "
                    f"(reference vs {category})"
                )

        return "Active"

    # ========================================================
    # IMPACT DIRECTION
    # ========================================================

    @staticmethod
    def _get_direction(
        contribution: float,
    ) -> str:
        """
        Determine whether a feature increases or decreases
        modeled hazard.

        Positive contribution:
            higher churn hazard

        Negative contribution:
            lower churn hazard
        """

        if contribution > 0:
            return "increases_risk"

        if contribution < 0:
            return "reduces_risk"

        return "neutral"

    # ========================================================
    # IMPACT LEVEL
    # ========================================================

    @staticmethod
    def _get_impact_level(
        absolute_contribution: float,
    ) -> str:
        """
        Convert mathematical contribution magnitude into a
        business-friendly impact category.

        These thresholds describe contribution magnitude,
        not statistical significance.
        """

        if absolute_contribution >= 0.75:
            return "high"

        if absolute_contribution >= 0.30:
            return "medium"

        return "low"

    # ========================================================
    # DRIVER DESCRIPTION
    # ========================================================

    def _build_description(
        self,
        feature_name: str,
        customer_input: Dict[str, Any],
        direction: str,
        hazard_multiplier: float,
    ) -> str:
        """
        Build a concise business-facing explanation.
        """

        label = self._get_base_feature_name(
            feature_name
        )

        value = self._get_feature_value_description(
            feature_name,
            customer_input,
        )

        multiplier = round(
            float(hazard_multiplier),
            3,
        )

        if direction == "increases_risk":

            if multiplier >= 1.0:

                return (
                    f"{label} = {value} "
                    f"is associated with a "
                    f"{multiplier}x modeled hazard "
                    f"relative to the Cox reference."
                )

            return (
                f"{label} = {value} "
                f"contributes to higher modeled "
                f"churn risk."
            )

        if direction == "reduces_risk":

            if multiplier > 0:

                return (
                    f"{label} = {value} "
                    f"is associated with a "
                    f"{multiplier}x modeled hazard "
                    f"relative to the Cox reference."
                )

            return (
                f"{label} = {value} "
                f"contributes to lower modeled "
                f"churn risk."
            )

        return (
            f"{label} = {value} "
            f"has minimal modeled impact."
        )

    # ========================================================
    # MAIN EXPLANATION METHOD
    # ========================================================

    def explain(
        self,
        processed_features: pd.DataFrame,
        customer_input: Dict[str, Any],
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate ranked feature-level explanations.

        Parameters
        ----------
        processed_features:
            Exactly the feature matrix supplied to the Cox model.

        customer_input:
            Original customer input.

        top_n:
            Maximum number of drivers returned per category.

        Returns
        -------
        dict
            Structured explainability information.
        """

        if not isinstance(
            processed_features,
            pd.DataFrame,
        ):
            raise TypeError(
                "processed_features must be a pandas DataFrame."
            )

        if processed_features.empty:
            raise ValueError(
                "processed_features cannot be empty."
            )

        if not isinstance(
            customer_input,
            dict,
        ):
            raise TypeError(
                "customer_input must be a dictionary."
            )

        if top_n < 1:
            raise ValueError(
                "top_n must be at least 1."
            )

        # ----------------------------------------------------
        # Use first customer row
        # ----------------------------------------------------

        feature_row = (
            processed_features.iloc[0]
        )

        contributions: List[Dict[str, Any]] = []

        # ----------------------------------------------------
        # Calculate coefficient contribution
        # ----------------------------------------------------

        for feature_name, coefficient in (
            self.coefficients.items()
        ):

            if feature_name not in feature_row.index:
                continue

            try:
                feature_value = float(
                    feature_row[feature_name]
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if not np.isfinite(
                feature_value
            ):
                continue

            coefficient = float(
                coefficient
            )

            contribution = (
                coefficient
                * feature_value
            )

            if not np.isfinite(
                contribution
            ):
                continue

            # ------------------------------------------------
            # Skip exact zero contribution
            # ------------------------------------------------

            if np.isclose(
                contribution,
                0.0,
                atol=1e-12,
            ):
                continue

            # ------------------------------------------------
            # Hazard multiplier generated by this feature
            # ------------------------------------------------

            try:
                hazard_multiplier = float(
                    np.exp(
                        np.clip(
                            contribution,
                            -20,
                            20,
                        )
                    )
                )

            except (
                OverflowError,
                FloatingPointError,
            ):
                continue

            direction = (
                self._get_direction(
                    contribution
                )
            )

            impact_level = (
                self._get_impact_level(
                    abs(contribution)
                )
            )

            label = (
                self._get_base_feature_name(
                    feature_name
                )
            )

            value = (
                self._get_feature_value_description(
                    feature_name,
                    customer_input,
                )
            )

            description = (
                self._build_description(
                    feature_name,
                    customer_input,
                    direction,
                    hazard_multiplier,
                )
            )

            contributions.append(
                {
                    "feature": label,
                    "model_feature": feature_name,
                    "value": value,
                    "coefficient": round(
                        coefficient,
                        6,
                    ),
                    "contribution": round(
                        contribution,
                        6,
                    ),
                    "hazard_multiplier": round(
                        hazard_multiplier,
                        3,
                    ),
                    "direction": direction,
                    "impact": impact_level,
                    "description": description,
                }
            )

        # ----------------------------------------------------
        # Sort by absolute contribution
        # ----------------------------------------------------

        contributions.sort(
            key=lambda item: abs(
                item["contribution"]
            ),
            reverse=True,
        )

        # ----------------------------------------------------
        # Split into risk and protective drivers
        # ----------------------------------------------------

        risk_drivers = [
            item
            for item in contributions
            if item["direction"]
            == "increases_risk"
        ][:top_n]

        protective_drivers = [
            item
            for item in contributions
            if item["direction"]
            == "reduces_risk"
        ][:top_n]

        # ----------------------------------------------------
        # All drivers
        # ----------------------------------------------------

        top_drivers = contributions[
            :top_n
        ]

        return {
            "top_drivers": top_drivers,
            "risk_drivers": risk_drivers,
            "protective_drivers": protective_drivers,
            "driver_count": len(
                contributions
            ),
        }