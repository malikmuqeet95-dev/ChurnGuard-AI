
"""
MLflow experiment tracking utilities.

Tracking is separate from production inference.
A tracking failure should not silently replace production artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


DEFAULT_EXPERIMENT_NAME = (
    "churnguard-cox-survival"
)


class ExperimentTracker:
    """Track model training runs using MLflow."""

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        tracking_uri: str | None = None,
    ) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(
                tracking_uri
            )

        self.experiment_name = experiment_name

        mlflow.set_experiment(
            self.experiment_name
        )

    def start_run(
        self,
        run_name: str | None = None,
    ):
        """Start an MLflow run."""

        return mlflow.start_run(
            run_name=run_name
        )

    @staticmethod
    def log_parameters(
        parameters: dict[str, Any],
    ) -> None:
        """Log training parameters."""

        normalized_parameters = {
            str(key): str(value)
            for key, value in parameters.items()
        }

        mlflow.log_params(
            normalized_parameters
        )

    @staticmethod
    def log_metrics(
        metrics: dict[str, float],
    ) -> None:
        """Log numerical training metrics."""

        normalized_metrics = {
            str(key): float(value)
            for key, value in metrics.items()
        }

        mlflow.log_metrics(
            normalized_metrics
        )

    @staticmethod
    def log_tags(
        tags: dict[str, str],
    ) -> None:
        """Log descriptive metadata as tags."""

        normalized_tags = {
            str(key): str(value)
            for key, value in tags.items()
        }

        mlflow.set_tags(
            normalized_tags
        )

    @staticmethod
    def log_artifact(
        artifact_path: str | Path,
    ) -> None:
        """Log a local file as an MLflow artifact."""

        path = Path(artifact_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Artifact does not exist: {path}"
            )

        mlflow.log_artifact(
            str(path)
        )

    @staticmethod
    def get_active_run_id() -> str | None:
        """Return the active MLflow run ID, if available."""

        active_run = mlflow.active_run()

        if active_run is None:
            return None

        return active_run.info.run_id