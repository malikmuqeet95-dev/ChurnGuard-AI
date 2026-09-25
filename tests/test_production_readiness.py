
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
APP_DIR = BACKEND_DIR / "app"
SRC_DIR = BACKEND_DIR / "src"


@pytest.mark.parametrize(
    "relative_path",
    [
        "backend/app/main.py",
        "backend/app/config.py",
        "backend/app/schemas.py",
        "backend/app/prediction_service.py",
        "backend/src/survival_predictor.py",
        "backend/src/model_quality_gate.py",
        "backend/src/model_promotion.py",
        "backend/src/drift_monitoring.py",
        "backend/src/mlops_monitoring_orchestrator.py",
        "frontend/index.html",
        "frontend/css/style.css",
        "frontend/js/app.js",
        "Dockerfile",
    ],
)
def test_required_production_files_exist(relative_path):
    file_path = PROJECT_ROOT / relative_path

    assert file_path.exists(), (
        f"Required project file is missing: {relative_path}"
    )

    assert file_path.is_file(), (
        f"Expected a file but found a different path: {relative_path}"
    )


def test_backend_and_source_directories_exist():
    assert BACKEND_DIR.is_dir()
    assert APP_DIR.is_dir()
    assert SRC_DIR.is_dir()


def test_production_environment_template_exists():
    production_env = PROJECT_ROOT / ".env.production.example"

    assert production_env.exists()
    assert production_env.is_file()


def test_production_environment_template_does_not_contain_real_secrets():
    production_env = PROJECT_ROOT / ".env.production.example"

    content = production_env.read_text(encoding="utf-8")

    forbidden_patterns = [
        "sk-",
        "ghp_",
        "-----BEGIN PRIVATE KEY-----",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in content, (
            f"Potential secret pattern found in production template: {pattern}"
        )


def test_python_cache_directories_are_not_required_project_artifacts():
    """
    This test documents that Python cache folders are not treated
    as required deployment files.
    """

    cache_directories = [
        PROJECT_ROOT / "__pycache__",
        BACKEND_DIR / "__pycache__",
        SRC_DIR / "__pycache__",
    ]

    for cache_directory in cache_directories:
        if cache_directory.exists():
            assert cache_directory.is_dir()