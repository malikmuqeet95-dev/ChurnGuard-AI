
"""
MLOps-Compatible Feature Engineering Pipeline.

This pipeline preserves the feature schema used by the existing
CoxPH production model.

It supports:
- Deterministic feature selection
- Numeric validation
- StandardScaler fitting/loading
- One-hot categorical encoding
- Target separation
- Feature schema tracking
- Training and inference modes

The pipeline does not overwrite the production model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_PIPELINE_VERSION = "1.1.0"

TARGET_COLUMNS = [
    "tenure_months",
    "churn_event",
]

CATEGORICAL_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_COLUMNS = [
    "SeniorCitizen",
]

MODEL_FEATURE_COLUMNS = [
    "SeniorCitizen",
    "MonthlyCharges_scaled",
    "TotalCharges_scaled",
]

DEFAULT_SCALER_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "processed"
    / "scaler.pkl"
)


@dataclass(frozen=True)
class FeaturePipelineResult:
    """Result returned by the feature pipeline."""

    features: pd.DataFrame
    target: pd.DataFrame
    feature_columns: tuple[str, ...]
    pipeline_version: str


class FeaturePipeline:
    """
    Feature pipeline compatible with the existing CoxPH model.

    Training mode:
        - Fits StandardScaler
        - Saves scaler
        - Encodes categorical values
        - Returns features and targets

    Inference mode:
        - Loads an existing StandardScaler
        - Applies the same numeric transformation
        - Returns features without requiring target columns
          when include_targets=False
    """

    def __init__(
        self,
        pipeline_version: str = DEFAULT_FEATURE_PIPELINE_VERSION,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
    ) -> None:
        self.pipeline_version = pipeline_version
        self.scaler_path = Path(scaler_path)

    
    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
        include_targets: bool,
    ) -> None:
        """
        Validate the minimum input requirements.

        Categorical columns and SeniorCitizen are optional for
        compatibility with minimal datasets and existing tests.

        The complete production dataset can still be validated
        separately before production training.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Input DataFrame is empty."
            )

        required_columns = set(
            NUMERIC_COLUMNS
        )

        if include_targets:
            required_columns |= set(
                TARGET_COLUMNS
            )

        missing_columns = (
            required_columns - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

    @staticmethod
    def _validate_targets(
        dataframe: pd.DataFrame,
    ) -> None:
        target = dataframe[TARGET_COLUMNS].copy()

        target["tenure_months"] = pd.to_numeric(
            target["tenure_months"],
            errors="coerce",
        )

        target["churn_event"] = pd.to_numeric(
            target["churn_event"],
            errors="coerce",
        )

        if target.isna().any().any():
            raise ValueError(
                "Target columns contain invalid or missing values."
            )

        if (target["tenure_months"] < 0).any():
            raise ValueError(
                "tenure_months cannot contain negative values."
            )

        if not target["churn_event"].isin([0, 1]).all():
            raise ValueError(
                "churn_event must contain only 0 or 1."
            )

    @staticmethod
    def _convert_numeric_columns(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        result = dataframe.copy()

        for column in NUMERIC_COLUMNS:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

        if result[NUMERIC_COLUMNS].isna().any().any():
            raise ValueError(
                "Numeric columns contain invalid or missing values."
            )

        if (
            result[NUMERIC_COLUMNS]
            .isin([float("inf"), float("-inf")])
            .any()
            .any()
        ):
            raise ValueError(
                "Numeric columns contain non-finite values."
            )

        if (result["MonthlyCharges"] <= 0).any():
            raise ValueError(
                "MonthlyCharges must be greater than zero."
            )

        if (result["TotalCharges"] < 0).any():
            raise ValueError(
                "TotalCharges cannot be negative."
            )

        return result

    
    @staticmethod
    def _validate_binary_columns(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate SeniorCitizen when present.

        Missing SeniorCitizen is allowed and will be
        filled with zero during feature preparation.
        """

        if "SeniorCitizen" not in dataframe.columns:
            return

        senior_citizen = pd.to_numeric(
            dataframe["SeniorCitizen"],
            errors="coerce",
        )

        if senior_citizen.isna().any():
            raise ValueError(
                "SeniorCitizen contains invalid or missing values."
            )

        if not senior_citizen.isin([0, 1]).all():
            raise ValueError(
                "SeniorCitizen must contain only 0 or 1."
            )

    def _scale_numeric_columns(
        self,
        dataframe: pd.DataFrame,
        is_training: bool,
    ) -> pd.DataFrame:
        if is_training:
            scaler = StandardScaler()

            scaled_values = scaler.fit_transform(
                dataframe[NUMERIC_COLUMNS]
            )

            self.scaler_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            joblib.dump(
                scaler,
                self.scaler_path,
            )

        else:
            if not self.scaler_path.exists():
                raise FileNotFoundError(
                    f"Scaler artifact not found: {self.scaler_path}"
                )

            scaler = joblib.load(self.scaler_path)

            scaled_values = scaler.transform(
                dataframe[NUMERIC_COLUMNS]
            )

        return pd.DataFrame(
            scaled_values,
            columns=[
                "MonthlyCharges_scaled",
                "TotalCharges_scaled",
            ],
            index=dataframe.index,
        )

    
    @staticmethod
    def _encode_categorical_columns(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Encode only categorical columns available in the input.

        Missing categorical columns are allowed for minimal
        datasets and are handled by the production validation
        layer when complete input is required.
        """

        available_columns = [
            column
            for column in CATEGORICAL_COLUMNS
            if column in dataframe.columns
        ]

        if not available_columns:
            return pd.DataFrame(
                index=dataframe.index
            )

        encoded = pd.get_dummies(
            dataframe[available_columns],
            drop_first=True,
            dtype=int,
        )

        return encoded

    def transform(
        self,
        dataframe: pd.DataFrame,
        is_training: bool = True,
        include_targets: bool = True,
    ) -> FeaturePipelineResult:
        """
        Transform a clean Telco DataFrame.

        Args:
            dataframe:
                Clean input dataset.

            is_training:
                If True, fit and save the scaler.
                If False, load the existing scaler.

            include_targets:
                If True, require and return survival targets.
                If False, return an empty target DataFrame.
        """

        self._validate_dataframe(
            dataframe,
            include_targets=include_targets,
        )

        working = dataframe.copy()

        if include_targets:
            self._validate_targets(working)

        self._validate_binary_columns(working)

        working = self._convert_numeric_columns(working)

        if include_targets:
            target = working[TARGET_COLUMNS].copy()
        else:
            target = pd.DataFrame(index=working.index)

        scaled_numeric = self._scale_numeric_columns(
            working,
            is_training=is_training,
        )

        
        if "SeniorCitizen" in working.columns:
            binary_features = (
                working[BINARY_COLUMNS]
                .astype(int)
                .reset_index(drop=True)
            )
        else:
            binary_features = pd.DataFrame(
                {
                    "SeniorCitizen": [0] * len(working)
                }
            )

        encoded_categorical = self._encode_categorical_columns(
            working
        ).reset_index(drop=True)

        scaled_numeric = scaled_numeric.reset_index(drop=True)

        features = pd.concat(
            [
                binary_features,
                scaled_numeric,
                encoded_categorical,
            ],
            axis=1,
        )

        features = features.apply(
            pd.to_numeric,
            errors="raise",
        )

        features = features.reset_index(drop=True)
        target = target.reset_index(drop=True)

        feature_columns = tuple(
            str(column)
            for column in features.columns
        )

        return FeaturePipelineResult(
            features=features,
            target=target,
            feature_columns=feature_columns,
            pipeline_version=self.pipeline_version,
        )