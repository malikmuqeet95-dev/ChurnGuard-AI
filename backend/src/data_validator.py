"""
MLOps Data Validation

Validates the processed Telco churn dataset before it enters
the machine-learning lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "tenure_months",
    "churn_event",
}


@dataclass(frozen=True)
class ValidationResult:
    """Structured result of dataset validation."""

    valid: bool
    row_count: int
    column_count: int
    errors: tuple[str, ...]

    @property
    def error_count(self) -> int:
        return len(self.errors)


class DataValidator:
    """
    Validates the processed churn dataset.

    The validator is intentionally independent from the
    prediction API so it can later be reused by training
    and scheduled MLOps pipelines.
    """

    def __init__(
        self,
        required_columns: set[str] | None = None,
    ) -> None:

        self.required_columns = (
            required_columns
            if required_columns is not None
            else REQUIRED_COLUMNS
        )

    def validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> ValidationResult:
        """Validate a pandas DataFrame."""

        errors: list[str] = []

        if dataframe.empty:
            errors.append(
                "Dataset is empty."
            )

        missing_columns = (
            self.required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            errors.append(
                "Missing required columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        if dataframe.columns.duplicated().any():
            duplicate_columns = (
                dataframe.columns[
                    dataframe.columns.duplicated()
                ]
                .tolist()
            )

            errors.append(
                "Duplicate columns: "
                + ", ".join(
                    map(str, duplicate_columns)
                )
            )

        if dataframe.isna().any().any():
            errors.append(
                "Dataset contains missing values."
            )

        numeric_columns = dataframe.select_dtypes(
            include=[np.number]
        ).columns

        if len(numeric_columns) > 0:

            numeric_values = dataframe[
                numeric_columns
            ].to_numpy(dtype=float)

            if not np.isfinite(
                numeric_values
            ).all():

                errors.append(
                    "Dataset contains "
                    "infinite or non-finite "
                    "numeric values."
                )

        if "tenure_months" in dataframe.columns:

            tenure = dataframe[
                "tenure_months"
            ]

            if not pd.api.types.is_numeric_dtype(
                tenure
            ):

                errors.append(
                    "tenure_months must be numeric."
                )

            else:

                if (tenure < 0).any():

                    errors.append(
                        "tenure_months contains "
                        "negative values."
                    )

        if "churn_event" in dataframe.columns:

            churn = dataframe[
                "churn_event"
            ]

            if not pd.api.types.is_numeric_dtype(
                churn
            ):

                errors.append(
                    "churn_event must be numeric."
                )

            else:

                allowed_values = {
                    0,
                    1,
                }

                observed_values = set(
                    churn.dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )

                invalid_values = (
                    observed_values
                    - allowed_values
                )

                if invalid_values:

                    errors.append(
                        "churn_event contains "
                        "invalid values: "
                        + ", ".join(
                            map(
                                str,
                                sorted(
                                    invalid_values
                                ),
                            )
                        )
                    )

        return ValidationResult(
            valid=len(errors) == 0,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            errors=tuple(errors),
        )

    def validate_file(
        self,
        path: str | Path,
    ) -> ValidationResult:
        """Load and validate a CSV dataset."""

        csv_path = Path(path)

        if not csv_path.is_file():

            return ValidationResult(
                valid=False,
                row_count=0,
                column_count=0,
                errors=(
                    f"Dataset file does not exist: "
                    f"{csv_path}",
                ),
            )

        try:

            dataframe = pd.read_csv(
                csv_path
            )

        except Exception as exc:

            return ValidationResult(
                valid=False,
                row_count=0,
                column_count=0,
                errors=(
                    f"Unable to read dataset: {exc}",
                ),
            )

        return self.validate_dataframe(
            dataframe
        )