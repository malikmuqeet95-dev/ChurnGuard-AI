
"""
Versioned model artifact registry.

This module stores trained model artifacts in version-specific directories.
It does not automatically promote a model to production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Metadata describing a versioned model artifact."""

    model_name: str
    model_version: str
    model_type: str
    artifact_file: str
    artifact_sha256: str
    created_at_utc: str
    penalizer: float
    random_state: int
    validation_fraction: float
    training_rows: int
    validation_rows: int
    training_c_index: float
    validation_c_index: float
    feature_columns: tuple[str, ...]
    dataset_version: str | None = None
    feature_pipeline_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata into a JSON-compatible dictionary."""

        result = asdict(self)
        result["feature_columns"] = list(self.feature_columns)
        return result


class ModelArtifactRegistry:
    """
    Manage versioned model artifacts.

    Example version:
        cox_ph_20260920T143000Z_a1b2c3d4
    """

    def __init__(
        self,
        registry_dir: str | Path,
    ) -> None:
        self.registry_dir = Path(registry_dir)

        self.versions_dir = (
            self.registry_dir / "versions"
        )

        self.manifest_path = (
            self.registry_dir / "model_manifest.json"
        )

        self.versions_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def calculate_sha256(
        file_path: str | Path,
    ) -> str:
        """Calculate the SHA-256 checksum of a file."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Artifact file does not exist: {path}"
            )

        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def create_model_version(
        model_name: str,
        artifact_sha256: str,
    ) -> str:
        """
        Create a deterministic-readable model version.

        The checksum suffix helps identify the artifact content.
        """

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        checksum_suffix = artifact_sha256[:8]

        safe_model_name = model_name.replace(
            " ",
            "_",
        )

        return (
            f"{safe_model_name}_"
            f"{timestamp}_"
            f"{checksum_suffix}"
        )

    def register_artifact(
        self,
        artifact_path: str | Path,
        metadata: ModelArtifactMetadata,
        evaluation: dict[str, Any] | None = None,
    ) -> Path:
        """
        Copy an artifact into its immutable version directory.

        Existing version directories are not overwritten.
        """

        source_path = Path(artifact_path)

        if not source_path.exists():
            raise FileNotFoundError(
                f"Artifact file does not exist: {source_path}"
            )

        version_dir = (
            self.versions_dir
            / metadata.model_name
            / metadata.model_version
        )

        if version_dir.exists():
            raise FileExistsError(
                "Model version already exists: "
                f"{metadata.model_version}"
            )

        version_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        destination_path = (
            version_dir / "model.pkl"
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        metadata_path = (
            version_dir / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata.to_dict(),
                file,
                indent=2,
            )

        if evaluation is not None:
            evaluation_path = (
                version_dir / "evaluation.json"
            )

            with evaluation_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    evaluation,
                    file,
                    indent=2,
                )

        return version_dir

    def update_manifest(
        self,
        model_name: str,
        model_version: str,
        metadata_path: str | Path,
    ) -> Path:
        """
        Update the registry manifest.

        This records the latest registered artifact.
        It does not automatically mark it as production-approved.
        """

        manifest: dict[str, Any] = {}

        if self.manifest_path.exists():
            with self.manifest_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                manifest = json.load(file)

        models = manifest.setdefault(
            "models",
            {},
        )

        models[model_name] = {
            "latest_registered_version": model_version,
            "metadata_path": str(metadata_path),
            "updated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        with self.manifest_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
            )

        return self.manifest_path

    def load_manifest(self) -> dict[str, Any]:
        """Load the registry manifest."""

        if not self.manifest_path.exists():
            return {}

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)