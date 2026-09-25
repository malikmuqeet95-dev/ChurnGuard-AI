"""
Phase 7 - Step 2
Data Validation Tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_validator import DataValidator


def valid_dataframe():
    return pd.DataFrame(
        {
            "tenure_months": [1, 12, 24],
            "churn_event": [0, 1, 0],
            "feature_a": [1.0, 2.0, 3.0],
        }
    )


def test_valid_dataframe_passes():

    validator = DataValidator()

    result = validator.validate_dataframe(
        valid_dataframe()
    )

    assert result.valid is True
    assert result.row_count == 3
    assert result.error_count == 0


def test_empty_dataframe_fails():

    validator = DataValidator()

    result = validator.validate_dataframe(
        pd.DataFrame()
    )

    assert result.valid is False
    assert "Dataset is empty." in result.errors


def test_missing_required_column_fails():

    validator = DataValidator()

    dataframe = valid_dataframe().drop(
        columns=["churn_event"]
    )

    result = validator.validate_dataframe(
        dataframe
    )

    assert result.valid is False
    assert any(
        "Missing required columns" in error
        for error in result.errors
    )


def test_duplicate_columns_fail():

    validator = DataValidator()

    dataframe = pd.DataFrame(
        [
            [1, 0, 10],
            [12, 1, 20],
        ],
        columns=[
            "tenure_months",
            "churn_event",
            "feature_a",
        ],
    )

    dataframe["feature_a_duplicate"] = (
        dataframe["feature_a"]
    )

    result = validator.validate_dataframe(
        dataframe
    )

    assert result.valid is True


def test_missing_values_fail():

    validator = DataValidator()

    dataframe = valid_dataframe()

    dataframe.loc[1, "feature_a"] = np.nan

    result = validator.validate_dataframe(
        dataframe
    )

    assert result.valid is False
    assert any(
        "missing values" in error.lower()
        for error in result.errors
    )


def test_infinite_values_fail():

    validator = DataValidator()

    dataframe = valid_dataframe()

    dataframe.loc[1, "feature_a"] = np.inf

    result = validator.validate_dataframe(
        dataframe
    )

    assert result.valid is False
    assert any(
        "non-finite" in error.lower()
        for error in result.errors
    )


def test_negative_tenure_fails():

    validator = DataValidator()

    dataframe = valid_dataframe()

    dataframe.loc[1, "tenure_months"] = -1

    result = validator.validate_dataframe(
        dataframe
    )

    assert result.valid is False
    assert any(
        "negative" in error.lower()
        for error in result.errors
    )


def test_invalid_churn_event_fails():

    validator = DataValidator()

    dataframe = valid_dataframe()

    dataframe.loc[1, "churn_event"] = 2

    result = validator.validate_dataframe(
        dataframe
    )

    assert result.valid is False
    assert any(
        "invalid values" in error.lower()
        for error in result.errors
    )


def test_validator_can_validate_real_processed_dataset():

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "backend"
        / "src"
        / "data"
        / "processed"
        / "telco_churn_features.csv"
    )

    validator = DataValidator()

    result = validator.validate_file(
        dataset_path
    )

    assert result.valid is True
    assert result.row_count > 0
    assert result.column_count > 0
    assert result.error_count == 0