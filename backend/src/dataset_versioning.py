"""
MLOps Dataset Versioning and Provenance

Creates deterministic fingerprints and metadata manifests
for datasets used in the machine-learning lifecycle.

This module does not modify the dataset.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCHEMA_VERSION = "1.0.0"


class DatasetVersionManager:
    """
    Creates and verifies dataset fingerprints and manifests.
    """

    def __init__(
        self,
        schema_version: str = DEFAULT_SCHEMA_VERSION,
    ) -> None:

        self.schema_version = schema_version

    @staticmethod
    def calculate_sha256(
        path: str | Path,
    ) -> str:
        """
        Calculate the SHA-256 hash of a file.

        The file is read in chunks to avoid loading large
        datasets entirely into memory.
        """

        file_path = Path(path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Dataset file does not exist: {file_path}"
            )

        digest = hashlib.sha256()

        with file_path.open("rb") as file:

            while True:

                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def calculate_dataframe_fingerprint(
        dataframe: pd.DataFrame,
    ) -> str:
        """
        Calculate a deterministic fingerprint for a DataFrame.

        The fingerprint includes:
        - column names
        - column order
        - data values
        """

        normalized = dataframe.copy()

        column_metadata = json.dumps(
            list(normalized.columns),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")

        row_values = pd.util.hash_pandas_object(
            normalized,
            index=True,
        ).to_numpy().tobytes()

        digest = hashlib.sha256()

        digest.update(column_metadata)
        digest.update(row_values)

        return digest.hexdigest()

    def create_manifest(
        self,
        path: str | Path,
        dataset_name: str = "telco_churn_features",
        source: str = "local_csv",
        feature_pipeline_version: str = "1.0.0",
    ) -> dict[str, Any]:
        """
        Create a metadata manifest for a CSV dataset.
        """

        file_path = Path(path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Dataset file does not exist: {file_path}"
            )

        dataframe = pd.read_csv(file_path)

        file_hash = self.calculate_sha256(
            file_path
        )

        dataframe_hash = (
            self.calculate_dataframe_fingerprint(
                dataframe
            )
        )

        manifest = {
            "dataset_name": dataset_name,
            "dataset_version": file_hash[:12],
            "schema_version": self.schema_version,
            "feature_pipeline_version": (
                feature_pipeline_version
            ),
            "source": source,
            "file_name": file_path.name,
            "file_sha256": file_hash,
            "dataframe_sha256": dataframe_hash,
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "columns": [
                str(column)
                for column in dataframe.columns
            ],
            "created_at_utc": (
                datetime.now(timezone.utc)
                .isoformat()
            ),
        }

        return manifest

    @staticmethod
    def save_manifest(
        manifest: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """
        Save a manifest as a JSON file.
        """

        destination = Path(output_path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=True,
            )

        return destination

    @staticmethod
    def load_manifest(
        manifest_path: str | Path,
    ) -> dict[str, Any]:
        """
        Load an existing JSON manifest.
        """

        path = Path(manifest_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Manifest file does not exist: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            manifest = json.load(file)

        if not isinstance(manifest, dict):
            raise ValueError(
                "Manifest must contain a JSON object."
            )

        return manifest

    def verify_manifest(
        self,
        path: str | Path,
        manifest: dict[str, Any],
    ) -> bool:
        """
        Verify that a dataset matches the recorded manifest.

        Returns True when both the file and DataFrame
        fingerprints match.
        """

        file_path = Path(path)

        if not file_path.is_file():
            return False

        dataframe = pd.read_csv(file_path)

        current_file_hash = self.calculate_sha256(
            file_path
        )

        current_dataframe_hash = (
            self.calculate_dataframe_fingerprint(
                dataframe
            )
        )

        expected_file_hash = manifest.get(
            "file_sha256"
        )

        expected_dataframe_hash = manifest.get(
            "dataframe_sha256"
        )

        return (
            current_file_hash == expected_file_hash
            and current_dataframe_hash
            == expected_dataframe_hash
        )