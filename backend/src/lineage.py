
"""
Dataset, feature-pipeline, and model lineage utilities.

Lineage metadata describes which inputs and configuration were used
to produce a model artifact. It does not prove model quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from backend.src.dataset_versioning import (
    DatasetVersionManager,
)


@dataclass(frozen=True)
class TrainingLineage:
    """Traceability information for a training run."""

    dataset_name: str
    dataset_version: str
    dataset_path: str
    feature_pipeline_version: str
    model_name: str
    model_version: str
    training_run_id: str | None
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible lineage metadata."""

        return asdict(self)

    def save_json(
        self,
        output_path: str | Path,
    ) -> Path:
        """Save lineage metadata as JSON."""

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


def create_training_lineage(
    dataset_path: str | Path,
    model_name: str,
    model_version: str,
    feature_pipeline_version: str = "1.0.0",
    training_run_id: str | None = None,
) -> TrainingLineage:
    """
    Create lineage metadata from a dataset file.

    The dataset manifest is generated from the actual file contents.
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist: {dataset_path}"
        )

    manager = DatasetVersionManager()

    manifest = manager.create_manifest(
        path=dataset_path,
        dataset_name="telco_churn_features",
        source="local_csv",
        feature_pipeline_version=(
            feature_pipeline_version
        ),
    )

    return TrainingLineage(
        dataset_name=str(
            manifest["dataset_name"]
        ),
        dataset_version=str(
            manifest["dataset_version"]
        ),
        dataset_path=str(
            dataset_path.resolve()
        ),
        feature_pipeline_version=(
            feature_pipeline_version
        ),
        model_name=model_name,
        model_version=model_version,
        training_run_id=training_run_id,
        created_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
    )