"""
Phase 6 - Step 1
QA Architecture & Baseline Tests

These tests establish a high-level QA baseline for the
ChurnGuard AI system.

They intentionally avoid changing application behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SRC_DIR = BACKEND_DIR / "src"

MODEL_PATH = SRC_DIR / "models" / "cox_ph_model.pkl"
METADATA_PATH = SRC_DIR / "models" / "model_metadata.json"
SCALER_PATH = SRC_DIR / "data" / "processed" / "scaler.pkl"
FEATURES_PATH = (
    SRC_DIR
    / "data"
    / "processed"
    / "telco_churn_features.csv"
)


def test_project_structure_exists():
    """Verify the major project directories exist."""

    assert PROJECT_ROOT.exists()
    assert BACKEND_DIR.exists()
    assert FRONTEND_DIR.exists()
    assert SRC_DIR.exists()


def test_model_artifacts_exist():
    """Verify required ML artifacts exist."""

    assert MODEL_PATH.exists()
    assert METADATA_PATH.exists()
    assert SCALER_PATH.exists()
    assert FEATURES_PATH.exists()


def test_backend_application_exists():
    """Verify the FastAPI application files exist."""

    assert (BACKEND_DIR / "app").exists()
    assert (BACKEND_DIR / "app" / "main.py").exists()
    assert (BACKEND_DIR / "app" / "schemas.py").exists()
    assert (BACKEND_DIR / "app" / "prediction_service.py").exists()


def test_decision_intelligence_modules_exist():
    """Verify the decision-intelligence layer exists."""

    required_modules = [
        "survival_predictor.py",
        "explainability.py",
        "business_rules.py",
        "intervention_simulator.py",
        "action_priority.py",
        "intervention_library.py",
        "intervention_roi.py",
        "retention_decision.py",
    ]

    for module in required_modules:
        assert (SRC_DIR / module).exists(), (
            f"Required module missing: {module}"
        )


def test_frontend_application_exists():
    """Verify the frontend application files exist."""

    assert (FRONTEND_DIR / "index.html").exists()
    assert (FRONTEND_DIR / "css" / "style.css").exists()
    assert (FRONTEND_DIR / "js" / "app.js").exists()


def test_test_suite_exists():
    """Verify the test suite itself is present."""

    tests_dir = PROJECT_ROOT / "tests"

    assert tests_dir.exists()

    test_files = list(tests_dir.glob("test_*.py"))

    assert len(test_files) > 0


@pytest.mark.parametrize(
    "module_name",
    [
        "backend.app.main",
        "backend.app.schemas",
        "backend.app.prediction_service",
    ],
)
def test_backend_modules_importable(module_name):
    """Verify core backend modules can be imported."""

    module = __import__(module_name, fromlist=["*"])

    assert module is not None

