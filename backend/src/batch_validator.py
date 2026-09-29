"""
Batch CSV validation for ChurnGuard AI v2.1.0.

The validator performs structural and customer-level validation before
rows are passed to the prediction engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .batch_schema import (
    ALLOWED_CATEGORIES,
    CUSTOMER_ID_COLUMN,
    FORECAST_HORIZON_MAX,
    FORECAST_HORIZON_MIN,
    MONTHLY_CHARGES_MAX,
    MONTHLY_CHARGES_MIN,
    REQUIRED_BATCH_COLUMNS,
    SENIOR_CITIZEN_VALUES,
    TENURE_MAX,
    TENURE_MIN,
    TOTAL_CHARGES_MIN,
)


@dataclass
class BatchValidationResult:
    """Result of validating a batch customer dataset."""

    valid: bool
    dataframe: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_rows: int = 0
    valid_rows: int = 0


class BatchValidationError(ValueError):
    """Raised when batch customer data fails validation."""


def validate_batch_dataframe(
    dataframe: pd.DataFrame,
) -> BatchValidationResult:
    """
    Validate a customer dataframe against the ChurnGuard batch contract.

    The function is intentionally non-mutating: the original dataframe
    is not modified.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(dataframe, pd.DataFrame):
        return BatchValidationResult(
            valid=False,
            errors=["Input must be a pandas DataFrame."],
        )

    df = dataframe.copy()
    total_rows = len(df)

    # 1. Normalize 'customerID' to 'Customer ID' if present
    if "customerID" in df.columns and CUSTOMER_ID_COLUMN not in df.columns:
        df = df.rename(columns={"customerID": CUSTOMER_ID_COLUMN})
    elif "CustomerID" in df.columns and CUSTOMER_ID_COLUMN not in df.columns:
        df = df.rename(columns={"CustomerID": CUSTOMER_ID_COLUMN})

    # ------------------------------------------------------------------
    # Empty dataset
    # ------------------------------------------------------------------

    if total_rows == 0:
        return BatchValidationResult(
            valid=False,
            dataframe=df,
            errors=["Batch dataset is empty."],
            total_rows=0,
            valid_rows=0,
        )

    # ------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_BATCH_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if errors:
        return BatchValidationResult(
            valid=False,
            dataframe=df,
            errors=errors,
            warnings=warnings,
            total_rows=total_rows,
            valid_rows=0,
        )

    # ------------------------------------------------------------------
    # Customer IDs
    # ------------------------------------------------------------------

    customer_ids = df[CUSTOMER_ID_COLUMN]

    if customer_ids.isna().any():
        rows = (
            customer_ids[customer_ids.isna()]
            .index.tolist()
        )
        errors.append(
            f"{CUSTOMER_ID_COLUMN} contains missing values "
            f"at rows: {rows[:10]}"
        )

    normalized_ids = customer_ids.astype(str).str.strip()

    empty_id_mask = normalized_ids.eq("") | normalized_ids.eq("nan")

    if empty_id_mask.any():
        rows = empty_id_mask[empty_id_mask].index.tolist()
        errors.append(
            f"{CUSTOMER_ID_COLUMN} contains empty values "
            f"at rows: {rows[:10]}"
        )

    duplicate_mask = normalized_ids.duplicated(keep=False)

    if duplicate_mask.any():
        duplicate_ids = sorted(
            normalized_ids[duplicate_mask].unique().tolist()
        )

        errors.append(
            "Duplicate customer IDs found: "
            + ", ".join(duplicate_ids[:10])
        )

    # ------------------------------------------------------------------
    # Numeric columns
    # ------------------------------------------------------------------

    numeric_columns = [
        "tenure",
        "forecast_horizon",
        "SeniorCitizen",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for column in numeric_columns:
        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_mask = converted.isna()

        if invalid_mask.any():
            rows = invalid_mask[invalid_mask].index.tolist()

            errors.append(
                f"{column} contains non-numeric or missing values "
                f"at rows: {rows[:10]}"
            )

    # Stop numeric range validation if conversion itself failed.
    if not any(
        error.startswith(column + " contains non-numeric")
        for column in numeric_columns
        for error in errors
    ):
        tenure = pd.to_numeric(df["tenure"])
        horizon = pd.to_numeric(df["forecast_horizon"])
        senior_citizen = pd.to_numeric(df["SeniorCitizen"])
        monthly_charges = pd.to_numeric(df["MonthlyCharges"])
        total_charges = pd.to_numeric(df["TotalCharges"])

        # --------------------------------------------------------------
        # Tenure
        # --------------------------------------------------------------

        invalid = ~tenure.between(
            TENURE_MIN,
            TENURE_MAX,
        )

        if invalid.any():
            rows = invalid[invalid].index.tolist()
            errors.append(
                f"tenure must be between {TENURE_MIN} and "
                f"{TENURE_MAX}; invalid rows: {rows[:10]}"
            )

        # --------------------------------------------------------------
        # Forecast horizon
        # --------------------------------------------------------------

        invalid = ~horizon.between(
            FORECAST_HORIZON_MIN,
            FORECAST_HORIZON_MAX,
        )

        if invalid.any():
            rows = invalid[invalid].index.tolist()
            errors.append(
                f"forecast_horizon must be between "
                f"{FORECAST_HORIZON_MIN} and "
                f"{FORECAST_HORIZON_MAX}; "
                f"invalid rows: {rows[:10]}"
            )

        # --------------------------------------------------------------
        # Senior citizen
        # --------------------------------------------------------------

        invalid = ~senior_citizen.isin(
            SENIOR_CITIZEN_VALUES
        )

        if invalid.any():
            rows = invalid[invalid].index.tolist()
            errors.append(
                "SeniorCitizen must contain only 0 or 1; "
                f"invalid rows: {rows[:10]}"
            )

        # --------------------------------------------------------------
        # Monthly charges
        # --------------------------------------------------------------

        invalid = ~monthly_charges.between(
            MONTHLY_CHARGES_MIN,
            MONTHLY_CHARGES_MAX,
        )

        if invalid.any():
            rows = invalid[invalid].index.tolist()
            errors.append(
                f"MonthlyCharges must be between "
                f"{MONTHLY_CHARGES_MIN} and "
                f"{MONTHLY_CHARGES_MAX}; "
                f"invalid rows: {rows[:10]}"
            )

        # --------------------------------------------------------------
        # Total charges
        # --------------------------------------------------------------

        invalid = total_charges < TOTAL_CHARGES_MIN

        if invalid.any():
            rows = invalid[invalid].index.tolist()
            errors.append(
                f"TotalCharges must be >= {TOTAL_CHARGES_MIN}; "
                f"invalid rows: {rows[:10]}"
            )

    # ------------------------------------------------------------------
    # Categorical columns
    # ------------------------------------------------------------------

    for column, allowed_values in ALLOWED_CATEGORIES.items():
        values = df[column]

        missing_mask = values.isna()

        if missing_mask.any():
            rows = missing_mask[missing_mask].index.tolist()

            errors.append(
                f"{column} contains missing values "
                f"at rows: {rows[:10]}"
            )

            continue

        normalized = values.astype(str).str.strip()

        invalid_mask = ~normalized.isin(allowed_values)

        if invalid_mask.any():
            invalid_values = sorted(
                normalized[invalid_mask].unique().tolist()
            )

            errors.append(
                f"{column} contains invalid values: "
                + ", ".join(invalid_values[:10])
            )

    # ------------------------------------------------------------------
    # Warnings for extra columns
    # ------------------------------------------------------------------

    extra_columns = [
        column
        for column in df.columns
        if column not in REQUIRED_BATCH_COLUMNS
    ]

    if extra_columns:
        warnings.append(
            "Extra columns will be ignored: "
            + ", ".join(extra_columns)
        )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    if errors:
        return BatchValidationResult(
            valid=False,
            dataframe=df,
            errors=errors,
            warnings=warnings,
            total_rows=total_rows,
            valid_rows=0,
        )

    # Normalize fields that downstream prediction expects.
    df[CUSTOMER_ID_COLUMN] = (
        df[CUSTOMER_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    for column in ALLOWED_CATEGORIES:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    return BatchValidationResult(
        valid=True,
        dataframe=df,
        errors=[],
        warnings=warnings,
        total_rows=total_rows,
        valid_rows=total_rows,
    )


def validate_batch_or_raise(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate a batch and return its normalized dataframe.

    Raises:
        BatchValidationError: if validation fails.
    """

    result = validate_batch_dataframe(dataframe)

    if not result.valid:
        message = "Batch validation failed:\n- " + "\n- ".join(
            result.errors
        )
        raise BatchValidationError(message)

    if result.dataframe is None:
        raise BatchValidationError(
            "Batch validation succeeded but produced no dataframe."
        )

    return result.dataframe


def summarize_validation(
    result: BatchValidationResult,
) -> dict[str, Any]:
    """Return a JSON-friendly validation summary."""

    return {
        "valid": result.valid,
        "total_rows": result.total_rows,
        "valid_rows": result.valid_rows,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "errors": result.errors,
        "warnings": result.warnings,
    }