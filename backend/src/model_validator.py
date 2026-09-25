from pathlib import Path
import joblib
import numpy as np


class ModelValidator:
    def __init__(
        self,
        model_path: str | Path,
        scaler_path: str | Path,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)

        self.model = None
        self.scaler = None

    def load_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found: {self.scaler_path}"
            )

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        return self.model, self.scaler

    def validate_artifacts(self):
        if self.model is None or self.scaler is None:
            self.load_artifacts()

        errors = []
        warnings = []

        if not hasattr(self.model, "predict_partial_hazard"):
            errors.append(
                "Loaded model does not provide "
                "predict_partial_hazard()."
            )

        if not hasattr(self.model, "predict_survival_function"):
            errors.append(
                "Loaded model does not provide "
                "predict_survival_function()."
            )

        if not hasattr(self.model, "params_"):
            errors.append(
                "Loaded Cox model does not contain params_."
            )

        if not hasattr(self.scaler, "transform"):
            errors.append(
                "Loaded scaler does not provide transform()."
            )

        model_feature_count = None

        if hasattr(self.model, "params_"):
            try:
                model_feature_count = len(
                    self.model.params_
                )
            except Exception:
                warnings.append(
                    "Could not determine model feature count."
                )

        scaler_feature_count = getattr(
            self.scaler,
            "n_features_in_",
            None,
        )

        if (
            model_feature_count is not None
            and scaler_feature_count is not None
            and model_feature_count != scaler_feature_count
        ):
            warnings.append(
                "Model/scaler feature-count mismatch: "
                f"model={model_feature_count}, "
                f"scaler={scaler_feature_count}. "
                "This is expected when the scaler is only used "
                "for the numeric inputs in the prediction pipeline."
            )

        if errors:
            raise RuntimeError(
                "Model validation failed:\n- "
                + "\n- ".join(errors)
            )

        return {
            "valid": True,
            "model_type": type(self.model).__name__,
            "scaler_type": type(self.scaler).__name__,
            "model_features": model_feature_count,
            "scaler_features": model_feature_count
            if model_feature_count is not None
            else scaler_feature_count,
            "scaler_input_features": scaler_feature_count,
            "warnings": warnings,
        }

    def validate_prediction_output(self, result: dict):
        required_keys = [
            "current_tenure_months",
            "forecast_horizon_months",
            "target_month",
            "hazard_ratio_multiplier",
            "projected_churn_pct",
            "projected_retention_pct",
            "risk_tier",
            "recommended_action",
            "survival_curve",
        ]

        missing = [
            key
            for key in required_keys
            if key not in result
        ]

        if missing:
            raise AssertionError(
                "Prediction result is missing keys: "
                + ", ".join(missing)
            )

        hazard = float(
            result["hazard_ratio_multiplier"]
        )

        churn = float(
            result["projected_churn_pct"]
        )

        retention = float(
            result["projected_retention_pct"]
        )

        if not np.isfinite(hazard):
            raise AssertionError(
                "Hazard ratio is not finite."
            )

        if hazard <= 0:
            raise AssertionError(
                "Hazard ratio must be greater than zero."
            )

        if not np.isfinite(churn):
            raise AssertionError(
                "Churn probability is not finite."
            )

        if not np.isfinite(retention):
            raise AssertionError(
                "Retention probability is not finite."
            )

        if not 0 <= churn <= 100:
            raise AssertionError(
                f"Churn probability out of range: {churn}"
            )

        if not 0 <= retention <= 100:
            raise AssertionError(
                "Retention probability out of range: "
                f"{retention}"
            )

        probability_sum = churn + retention

        if not np.isclose(
            probability_sum,
            100.0,
            atol=0.2,
        ):
            raise AssertionError(
                "Churn + retention should equal approximately "
                f"100%. Got {probability_sum:.4f}%."
            )

        curve = result["survival_curve"]

        if not isinstance(curve, list):
            raise AssertionError(
                "survival_curve must be a list."
            )

        if len(curve) == 0:
            raise AssertionError(
                "survival_curve cannot be empty."
            )

        previous_month = None

        for point in curve:
            if not isinstance(point, dict):
                raise AssertionError(
                    "Each survival curve point must be a dictionary."
                )

            for key in (
                "month",
                "retention",
                "churn",
            ):
                if key not in point:
                    raise AssertionError(
                        f"Survival curve point missing '{key}'."
                    )

            month = float(point["month"])
            curve_retention = float(
                point["retention"]
            )
            curve_churn = float(
                point["churn"]
            )

            if previous_month is not None:
                if month <= previous_month:
                    raise AssertionError(
                        "Survival curve months must be strictly increasing."
                    )

            previous_month = month

            if not 0 <= curve_retention <= 1:
                raise AssertionError(
                    "Curve retention must be between 0 and 1."
                )

            if not 0 <= curve_churn <= 1:
                raise AssertionError(
                    "Curve churn must be between 0 and 1."
                )

            if not np.isclose(
                curve_retention + curve_churn,
                1.0,
                atol=0.01,
            ):
                raise AssertionError(
                    "Curve retention + churn must equal "
                    "approximately 1."
                )

        return True
