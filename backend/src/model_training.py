
"""
Reproducible CoxPH model training pipeline.

This module:
- Runs log-rank analysis
- Performs reproducible train/validation splitting
- Trains a penalized CoxPH model
- Evaluates training and validation C-index
- Saves the model artifact
- Saves evaluation metadata

The existing production model is not overwritten unless the caller
explicitly chooses the production save path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

from src.model_evaluation import (
    ModelEvaluationResult,
    evaluate_cox_model,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(levelname)s - %(message)s"
    ),
)

BASE_DIR = Path(__file__).resolve().parent

FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_churn_features.csv"
)

CLEAN_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_churn_clean.csv"
)

MODEL_DIR = BASE_DIR / "models"

COX_MODEL_PATH = (
    MODEL_DIR / "cox_ph_model.pkl"
)

EVALUATION_PATH = (
    MODEL_DIR / "model_evaluation.json"
)

DEFAULT_PENALIZER = 0.05
DEFAULT_RANDOM_STATE = 42
DEFAULT_VALIDATION_FRACTION = 0.20


def run_logrank_analysis(
    clean_data_path: str | Path = CLEAN_DATA_PATH,
) -> float:
    """Perform a log-rank test across contract cohorts."""

    logging.info(
        "Running log-rank significance test."
    )

    df = pd.read_csv(clean_data_path)

    required_columns = {
        "Contract",
        "tenure_months",
        "churn_event",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    m2m = df[
        df["Contract"] == "Month-to-month"
    ]

    two_year = df[
        df["Contract"] == "Two year"
    ]

    if m2m.empty or two_year.empty:
        raise ValueError(
            "Both contract groups must contain records."
        )

    results = logrank_test(
        durations_A=m2m["tenure_months"],
        durations_B=two_year["tenure_months"],
        event_observed_A=m2m["churn_event"],
        event_observed_B=two_year["churn_event"],
    )

    logging.info(
        "Log-rank p-value: %.4e",
        results.p_value,
    )

    return float(results.p_value)


def _validate_training_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the engineered training dataset."""

    if dataframe.empty:
        raise ValueError(
            "Training feature DataFrame is empty."
        )

    required_columns = {
        "tenure_months",
        "churn_event",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required training columns: "
            + ", ".join(sorted(missing_columns))
        )

    if dataframe.isna().any().any():
        raise ValueError(
            "Training dataset contains missing values."
        )

    if not dataframe["churn_event"].isin([0, 1]).all():
        raise ValueError(
            "churn_event must contain only 0 or 1."
        )

    if (
        dataframe["tenure_months"] < 0
    ).any():
        raise ValueError(
            "tenure_months cannot contain negative values."
        )


def train_cox_model(
    features_path: str | Path = FEATURES_PATH,
    save_path: str | Path = COX_MODEL_PATH,
    penalizer: float = DEFAULT_PENALIZER,
    random_state: int = DEFAULT_RANDOM_STATE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    evaluation_path: str | Path = EVALUATION_PATH,
    save_model: bool = True,
) -> tuple[CoxPHFitter, ModelEvaluationResult]:
    """
    Train and evaluate a CoxPH model.

    Returns:
        A tuple containing:
        - Fitted CoxPHFitter
        - Structured evaluation result
    """

    if penalizer < 0:
        raise ValueError(
            "penalizer cannot be negative."
        )

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    features_path = Path(features_path)
    save_path = Path(save_path)
    evaluation_path = Path(evaluation_path)

    logging.info(
        "Loading engineered features from %s",
        features_path,
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}"
        )

    dataframe = pd.read_csv(features_path)

    _validate_training_dataframe(dataframe)

    train_fraction = 1.0 - validation_fraction

    train_df = dataframe.sample(
        frac=train_fraction,
        random_state=random_state,
    )

    validation_df = dataframe.drop(
        train_df.index
    )

    if train_df.empty or validation_df.empty:
        raise ValueError(
            "Training and validation sets must not be empty."
        )

    logging.info(
        "Training rows: %d | Validation rows: %d",
        len(train_df),
        len(validation_df),
    )

    model = CoxPHFitter(
        penalizer=penalizer,
    )

    logging.info(
        "Fitting CoxPH model with penalizer %.4f",
        penalizer,
    )

    model.fit(
        train_df,
        duration_col="tenure_months",
        event_col="churn_event",
    )

    evaluation = evaluate_cox_model(
        model=model,
        train_df=train_df,
        validation_df=validation_df,
        penalizer=penalizer,
        random_state=random_state,
        validation_fraction=validation_fraction,
    )

    logging.info(
        "Training C-index: %.4f",
        evaluation.training_c_index,
    )

    logging.info(
        "Validation C-index: %.4f",
        evaluation.validation_c_index,
    )

    if save_model:
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            save_path,
        )

        logging.info(
            "Model artifact saved to %s",
            save_path,
        )

    evaluation.save_json(
        evaluation_path
    )

    logging.info(
        "Evaluation metadata saved to %s",
        evaluation_path,
    )

    return model, evaluation


if __name__ == "__main__":
    run_logrank_analysis()

    train_cox_model(
        save_model=True
    )