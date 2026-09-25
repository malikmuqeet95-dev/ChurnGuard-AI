
from __future__ import annotations

import json

import pandas as pd

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_FEATURES_PATH = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "data"
    / "processed"
    / "telco_churn_features.csv"
)

from backend.src.training_orchestrator import (
    run_versioned_training,
)



def test_versioned_training_registers_model_without_overwriting_production(
    tmp_path,
):
    features_path = tmp_path / "features.csv"

    shutil.copy2(
        REAL_FEATURES_PATH,
        features_path,
    )

    production_model_path = (
        tmp_path / "production_model.pkl"
    )

    production_model_path.write_bytes(
        b"original-production-model"
    )

    result = run_versioned_training(
        features_path=features_path,
        registry_path=tmp_path / "registry",
    )

    assert result["model_name"] == "cox_ph"
    assert result["model_version"].startswith(
        "cox_ph_"
    )

    version_directory = (
        tmp_path
        / "registry"
        / "versions"
        / "cox_ph"
        / result["model_version"]
    )

    assert version_directory.exists()
    assert (
        version_directory / "model.pkl"
    ).exists()
    assert (
        version_directory / "metadata.json"
    ).exists()
    assert (
        version_directory / "evaluation.json"
    ).exists()

    assert (
        production_model_path.read_bytes()
        == b"original-production-model"
    )


def test_versioned_training_contains_evaluation_metadata(
    tmp_path,
):
    features_path = tmp_path / "features.csv"

    shutil.copy2(
        REAL_FEATURES_PATH,
        features_path,
    )

    result = run_versioned_training(
        features_path=features_path,
        registry_path=tmp_path / "registry",
    )

    version_directory = (
        tmp_path
        / "registry"
        / "versions"
        / "cox_ph"
        / result["model_version"]
    )

    with (
        version_directory / "metadata.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["model_type"] == "CoxPHFitter"
    assert metadata["penalizer"] == 0.05
    assert metadata["random_state"] == 42
    assert metadata["validation_fraction"] == 0.20
    assert metadata["artifact_sha256"] == (
        result["artifact_sha256"]
    )