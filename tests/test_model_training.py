
import os

from lifelines import CoxPHFitter

from src.model_training import (
    COX_MODEL_PATH,
    EVALUATION_PATH,
    run_logrank_analysis,
    train_cox_model,
)


def test_logrank_significance():
    p_value = run_logrank_analysis()

    assert p_value < 0.05, (
        f"Log-rank p-value ({p_value}) "
        "is not statistically significant."
    )


def test_model_training_and_c_index():
    model, evaluation = train_cox_model()

    assert os.path.exists(
        COX_MODEL_PATH
    ), "Model artifact was not serialized."

    assert isinstance(
        model,
        CoxPHFitter,
    ), "Trained object is not a CoxPHFitter instance."

    assert model.concordance_index_ >= 0.70, (
        "Training concordance index is too low: "
        f"{model.concordance_index_:.4f}"
    )

    assert evaluation.training_c_index >= 0.70, (
        "Training C-index is too low: "
        f"{evaluation.training_c_index:.4f}"
    )

    assert 0.0 <= evaluation.validation_c_index <= 1.0

    assert evaluation.training_rows > 0
    assert evaluation.validation_rows > 0

    hazard_ratios = model.hazard_ratios_

    assert not hazard_ratios.isnull().any(), (
        "Hazard ratios contain NaN values."
    )


def test_evaluation_metadata_is_saved():
    _, evaluation = train_cox_model()

    assert os.path.exists(
        EVALUATION_PATH
    ), "Evaluation metadata was not saved."

    assert evaluation.model_type == "CoxPHFitter"
    assert evaluation.penalizer == 0.05
    assert evaluation.random_state == 42
    assert evaluation.validation_fraction == 0.20