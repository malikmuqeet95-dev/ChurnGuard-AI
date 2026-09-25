"""
Phase 7 - Step 4 & 5
Reproducible Feature Pipeline Tests (Aligned with Production Schema)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_pipeline import FeaturePipeline


def create_valid_dataframe() -> pd.DataFrame:
    """Create a test DataFrame containing all production Telco columns."""
    return pd.DataFrame(
        {
            "tenure_months": [1, 12, 24],
            "churn_event": [0, 1, 0],
            "SeniorCitizen": [0, 1, 0],
            "MonthlyCharges": [50.0, 75.0, 100.0],
            "TotalCharges": [50.0, 900.0, 2400.0],
            "gender": ["Male", "Female", "Male"],
            "Partner": ["No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes"],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
        }
    )


def test_pipeline_returns_features_and_target(tmp_path):
    pipeline = FeaturePipeline(scaler_path=tmp_path / "scaler.pkl")
    result = pipeline.transform(create_valid_dataframe(), is_training=True)

    assert isinstance(result.features, pd.DataFrame)
    assert isinstance(result.target, pd.DataFrame)
    assert len(result.features) == 3
    assert len(result.target) == 3


def test_target_columns_are_separated(tmp_path):
    pipeline = FeaturePipeline(scaler_path=tmp_path / "scaler.pkl")
    result = pipeline.transform(create_valid_dataframe(), is_training=True)

    assert list(result.target.columns) == ["tenure_months", "churn_event"]
    assert "tenure_months" not in result.features.columns
    assert "churn_event" not in result.features.columns


def test_categorical_features_are_encoded(tmp_path):
    pipeline = FeaturePipeline(scaler_path=tmp_path / "scaler.pkl")
    result = pipeline.transform(create_valid_dataframe(), is_training=True)

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in result.features.dtypes
    )
    assert "MonthlyCharges_scaled" in result.features.columns
    assert "TotalCharges_scaled" in result.features.columns


def test_pipeline_is_deterministic(tmp_path):
    dataframe = create_valid_dataframe()
    pipeline = FeaturePipeline(scaler_path=tmp_path / "scaler.pkl")

    first = pipeline.transform(dataframe, is_training=True)
    second = pipeline.transform(dataframe, is_training=False)

    pd.testing.assert_frame_equal(first.features, second.features)
    pd.testing.assert_frame_equal(first.target, second.target)
    assert first.feature_columns == second.feature_columns


def test_pipeline_does_not_mutate_input(tmp_path):
    dataframe = create_valid_dataframe()
    original = dataframe.copy(deep=True)

    pipeline = FeaturePipeline(scaler_path=tmp_path / "scaler.pkl")
    pipeline.transform(dataframe, is_training=True)

    pd.testing.assert_frame_equal(dataframe, original)


def test_pipeline_version_is_recorded(tmp_path):
    pipeline = FeaturePipeline(
        pipeline_version="2.1.0",
        scaler_path=tmp_path / "scaler.pkl",
    )
    result = pipeline.transform(create_valid_dataframe(), is_training=True)

    assert result.pipeline_version == "2.1.0"


def test_optional_categorical_column_can_be_absent(tmp_path):
    dataframe = create_valid_dataframe().drop(columns=["Contract"])
    pipeline = FeaturePipeline(scaler_path=tmp_path / "scaler.pkl")

    result = pipeline.transform(dataframe, is_training=True)
    assert isinstance(result.features, pd.DataFrame)
    assert "Contract_One year" not in result.features.columns
    assert "Contract_Two year" not in result.features.columns


def test_missing_target_column_is_rejected(tmp_path):
    dataframe = create_valid_dataframe().drop(columns=["churn_event"])
    pipeline = FeaturePipeline(scaler_path=tmp_path / "scaler.pkl")

    with pytest.raises(ValueError, match="Missing required columns.*churn_event"):
        pipeline.transform(dataframe, is_training=True, include_targets=True)


def test_invalid_numeric_value_is_rejected(tmp_path):
    dataframe = create_valid_dataframe()
    # Cast to object first to prevent pandas dtype deprecation warning
    dataframe["MonthlyCharges"] = dataframe["MonthlyCharges"].astype(object)
    dataframe.loc[0, "MonthlyCharges"] = "invalid"

    pipeline = FeaturePipeline(scaler_path=tmp_path / "scaler.pkl")

    with pytest.raises(ValueError):
        pipeline.transform(dataframe, is_training=True)


def test_empty_dataframe_is_rejected(tmp_path):
    dataframe = create_valid_dataframe().iloc[0:0]
    pipeline = FeaturePipeline(scaler_path=tmp_path / "scaler.pkl")

    with pytest.raises(ValueError, match="Input DataFrame is empty."):
        pipeline.transform(dataframe, is_training=True)