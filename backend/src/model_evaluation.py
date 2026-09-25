
"""
Model evaluation utilities for the CoxPH survival model.

This module evaluates training and holdout validation performance.
It does not save or replace production model artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json

import pandas as pd
from lifelines import CoxPHFitter


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Structured evaluation result."""

    model_type: str
    penalizer: float
    random_state: int
    validation_fraction: float
    training_rows: int
    validation_rows: int
    training_c_index: float
    validation_c_index: float
    feature_columns: tuple[str, ...]
    evaluated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation result to a JSON-compatible dictionary."""
        result = asdict(self)
        result["feature_columns"] = list(
            self.feature_columns
        )
        return result

    def save_json(
        self,
        output_path: str | Path,
    ) -> Path:
        """Save evaluation result as JSON."""
        path = Path(output_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
            )

        return path


def evaluate_cox_model(
    model: CoxPHFitter,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    penalizer: float,
    random_state: int,
    validation_fraction: float,
) -> ModelEvaluationResult:
    """
    Evaluate a fitted CoxPH model.

    The target columns are excluded from the feature-column metadata.
    """

    training_c_index = float(
        model.concordance_index_
    )

    validation_c_index = float(
        model.score(
            validation_df,
            scoring_method="concordance_index",
        )
    )

    feature_columns = tuple(
        str(column)
        for column in model.params_.index
    )

    return ModelEvaluationResult(
        model_type=type(model).__name__,
        penalizer=float(penalizer),
        random_state=int(random_state),
        validation_fraction=float(validation_fraction),
        training_rows=int(len(train_df)),
        validation_rows=int(len(validation_df)),
        training_c_index=training_c_index,
        validation_c_index=validation_c_index,
        feature_columns=feature_columns,
        evaluated_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
    )