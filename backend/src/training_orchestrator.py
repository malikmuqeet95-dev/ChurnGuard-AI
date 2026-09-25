
"""
Safe training orchestration for ChurnGuard AI.

This workflow creates and registers experimental model artifacts.
It does not overwrite the production inference model.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any
from backend.src.experiment_tracking import (
    ExperimentTracker,
)
from backend.src.model_registry import (
    ModelArtifactMetadata,
    ModelArtifactRegistry,
)
from backend.src.model_training import (
    train_cox_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FEATURES_PATH = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "data"
    / "processed"
    / "telco_churn_features.csv"
)

DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "models"
    / "registry"
)


def _evaluation_to_dict(
    evaluation: Any,
) -> dict[str, Any]:
    """Convert an evaluation object to a dictionary."""

    if hasattr(evaluation, "to_dict"):
        return evaluation.to_dict()

    if hasattr(evaluation, "__dataclass_fields__"):
        result = asdict(evaluation)

        if isinstance(
            result.get("feature_columns"),
            tuple,
        ):
            result["feature_columns"] = list(
                result["feature_columns"]
            )

        return result

    if isinstance(evaluation, dict):
        return dict(evaluation)

    raise TypeError(
        "Unsupported evaluation result type: "
        f"{type(evaluation).__name__}"
    )


def run_versioned_training(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    model_name: str = "cox_ph",
    penalizer: float = 0.05,
    random_state: int = 42,
    validation_fraction: float = 0.20,
    dataset_version: str | None = None,
    feature_pipeline_version: str = "1.0.0",
) -> dict[str, Any]:
    """
    Train and register an isolated model artifact.

    The existing production model is not overwritten.
    """

    features_path = Path(features_path)
    registry_path = Path(registry_path)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file does not exist: {features_path}"
        )

    registry = ModelArtifactRegistry(
        registry_dir=registry_path
    )

    with tempfile.TemporaryDirectory(
        prefix="churnguard_training_"
    ) as temporary_directory:

        temporary_directory_path = Path(
            temporary_directory
        )

        experimental_model_path = (
            temporary_directory_path
            / "experimental_model.pkl"
        )

        evaluation_path = (
            temporary_directory_path
            / "evaluation.json"
        )

        model, evaluation = train_cox_model(
            features_path=features_path,
            save_path=experimental_model_path,
            penalizer=penalizer,
            random_state=random_state,
            validation_fraction=validation_fraction,
            evaluation_path=evaluation_path,
            save_model=True,
        )

        evaluation_dict = _evaluation_to_dict(
            evaluation
        )

        artifact_sha256 = (
            registry.calculate_sha256(
                experimental_model_path
            )
        )

        model_version = (
            registry.create_model_version(
                model_name=model_name,
                artifact_sha256=artifact_sha256,
            )
        )

        metadata = ModelArtifactMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type=type(model).__name__,
            artifact_file="model.pkl",
            artifact_sha256=artifact_sha256,
            created_at_utc=datetime.now(
                timezone.utc
            ).isoformat(),
            penalizer=penalizer,
            random_state=random_state,
            validation_fraction=validation_fraction,
            training_rows=int(
                evaluation_dict["training_rows"]
            ),
            validation_rows=int(
                evaluation_dict["validation_rows"]
            ),
            training_c_index=float(
                evaluation_dict["training_c_index"]
            ),
            validation_c_index=float(
                evaluation_dict["validation_c_index"]
            ),
            feature_columns=tuple(
                evaluation_dict["feature_columns"]
            ),
            dataset_version=dataset_version,
            feature_pipeline_version=(
                feature_pipeline_version
            ),
        )

        version_directory = (
            registry.register_artifact(
                artifact_path=experimental_model_path,
                metadata=metadata,
                evaluation=evaluation_dict,
            )
        )

        metadata_path = (
            version_directory / "metadata.json"
        )

        manifest_path = (
            registry.update_manifest(
                model_name=model_name,
                model_version=model_version,
                metadata_path=metadata_path,
            )
        )

        return {
            "model_name": model_name,
            "model_version": model_version,
            "version_directory": str(
                version_directory
            ),
            "manifest_path": str(
                manifest_path
            ),
            "artifact_sha256": artifact_sha256,
            "training_c_index": float(
                evaluation_dict["training_c_index"]
            ),
            "validation_c_index": float(
                evaluation_dict["validation_c_index"]
            ),
            "dataset_version": dataset_version,
            "feature_pipeline_version": (
                feature_pipeline_version
            ),
        }


def run_tracked_versioned_training(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    tracking_uri: str | None = None,
    experiment_name: str = "churnguard-cox-survival",
    model_name: str = "cox_ph",
    penalizer: float = 0.05,
    random_state: int = 42,
    validation_fraction: float = 0.20,
    dataset_version: str | None = None,
    feature_pipeline_version: str = "1.0.0",
) -> dict[str, Any]:
    """
    Train, register, and track an experimental model.

    The production artifact is not overwritten.
    """

    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    with tracker.start_run(
        run_name="versioned-cox-training"
    ):
        result = run_versioned_training(
            features_path=features_path,
            registry_path=registry_path,
            model_name=model_name,
            penalizer=penalizer,
            random_state=random_state,
            validation_fraction=validation_fraction,
            dataset_version=dataset_version,
            feature_pipeline_version=(
                feature_pipeline_version
            ),
        )

        tracker.log_parameters(
            {
                "model_name": model_name,
                "penalizer": penalizer,
                "random_state": random_state,
                "validation_fraction": (
                    validation_fraction
                ),
                "feature_pipeline_version": (
                    feature_pipeline_version
                ),
                "dataset_version": (
                    dataset_version or "unknown"
                ),
            }
        )

        tracker.log_metrics(
            {
                "training_c_index": (
                    result["training_c_index"]
                ),
                "validation_c_index": (
                    result["validation_c_index"]
                ),
            }
        )

        tracker.log_tags(
            {
                "model_type": "CoxPHFitter",
                "model_version": (
                    result["model_version"]
                ),
                "workflow": (
                    "versioned_training"
                ),
            }
        )

        metadata_path = (
            Path(result["version_directory"])
            / "metadata.json"
        )

        evaluation_path = (
            Path(result["version_directory"])
            / "evaluation.json"
        )

        tracker.log_artifact(
            metadata_path
        )

        tracker.log_artifact(
            evaluation_path
        )

        result["mlflow_run_id"] = (
            tracker.get_active_run_id()
        )

        return result
    