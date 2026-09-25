"""
Phase 7 - Step 1
MLOps Foundation Audit

Verifies that the project contains the minimum artifacts
and components required for a reproducible ML lifecycle.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BACKEND_DIR = PROJECT_ROOT / "backend"
SRC_DIR = BACKEND_DIR / "src"

DATA_DIR = SRC_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = SRC_DIR / "models"

TESTS_DIR = PROJECT_ROOT / "tests"


def test_project_ml_directories_exist():
    """Required ML directories must exist."""

    assert BACKEND_DIR.is_dir()
    assert SRC_DIR.is_dir()
    assert DATA_DIR.is_dir()
    assert PROCESSED_DIR.is_dir()
    assert MODEL_DIR.is_dir()


def test_processed_dataset_exists():
    """The processed feature dataset must exist."""

    features_path = (
        PROCESSED_DIR
        / "telco_churn_features.csv"
    )

    assert features_path.is_file()


def test_scaler_artifact_exists():
    """The preprocessing scaler artifact must exist."""

    scaler_path = (
        PROCESSED_DIR
        / "scaler.pkl"
    )

    assert scaler_path.is_file()


def test_model_artifact_exists():
    """The trained Cox model artifact must exist."""

    model_path = (
        MODEL_DIR
        / "cox_ph_model.pkl"
    )

    assert model_path.is_file()


def test_model_metadata_exists():
    """Model metadata must exist."""

    metadata_path = (
        MODEL_DIR
        / "model_metadata.json"
    )

    assert metadata_path.is_file()


def test_survival_predictor_exists():
    """The production prediction engine must exist."""

    predictor_path = (
        SRC_DIR
        / "survival_predictor.py"
    )

    assert predictor_path.is_file()


def test_feature_engineering_tests_exist():
    """Feature-engineering validation must exist."""

    test_path = (
        TESTS_DIR
        / "test_feature_engineering.py"
    )

    assert test_path.is_file()


def test_model_training_tests_exist():
    """Model-training validation must exist."""

    test_path = (
        TESTS_DIR
        / "test_model_training.py"
    )

    assert test_path.is_file()


def test_prediction_engine_tests_exist():
    """Prediction-engine validation must exist."""

    test_path = (
        TESTS_DIR
        / "test_prediction_engine.py"
    )

    assert test_path.is_file()


def test_mlops_has_reproducible_artifact_chain():
    """
    Verify the core artifact chain exists:

    processed data
        ↓
    scaler
        ↓
    model
        ↓
    metadata
    """

    artifacts = [
        PROCESSED_DIR / "telco_churn_features.csv",
        PROCESSED_DIR / "scaler.pkl",
        MODEL_DIR / "cox_ph_model.pkl",
        MODEL_DIR / "model_metadata.json",
    ]

    assert all(
        artifact.is_file()
        for artifact in artifacts
    )


def test_existing_ml_modules_are_importable():
    """
    Existing ML components should remain importable.
    """

    from src.survival_predictor import SurvivalPredictor

    assert SurvivalPredictor is not None