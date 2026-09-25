
from __future__ import annotations

import shutil
from pathlib import Path

import mlflow

from backend.src.training_orchestrator import (
    run_tracked_versioned_training,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_FEATURES_PATH = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "data"
    / "processed"
    / "telco_churn_features.csv"
)


def test_tracked_training_logs_model_metrics(
    tmp_path,
):
    features_path = tmp_path / "features.csv"

    shutil.copy2(
        REAL_FEATURES_PATH,
        features_path,
    )

    db_path = (tmp_path / "mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{db_path}"

    result = (
        run_tracked_versioned_training(
            features_path=features_path,
            registry_path=tmp_path / "registry",
            tracking_uri=tracking_uri,
            experiment_name=(
                "test-integrated-training"
            ),
        )
    )

    assert result["mlflow_run_id"] is not None

    client = mlflow.MlflowClient(
        tracking_uri=tracking_uri
    )

    run = client.get_run(
        result["mlflow_run_id"]
    )

    assert (
        run.data.params["penalizer"]
        == "0.05"
    )

    assert (
        run.data.params["random_state"]
        == "42"
    )

    assert (
        "training_c_index"
        in run.data.metrics
    )

    assert (
        "validation_c_index"
        in run.data.metrics
    )

    assert (
        run.data.tags["model_type"]
        == "CoxPHFitter"
    )

    assert (
        run.data.tags["model_version"]
        == result["model_version"]
    )