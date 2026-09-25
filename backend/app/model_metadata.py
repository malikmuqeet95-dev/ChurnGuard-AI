from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.config import settings


class ModelMetadataService:
    """
    Loads and exposes metadata describing the active ML model.
    """

    def __init__(
        self,
        metadata_path: Path | None = None,
    ) -> None:
        self.metadata_path = (
            metadata_path
            if metadata_path is not None
            else settings.model_metadata_path
        )

        self._metadata: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        """
        Load model metadata from JSON.

        Metadata is cached after the first successful load.
        """

        if self._metadata is not None:
            return self._metadata

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata file not found: {self.metadata_path}"
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Model metadata path is not a file: {self.metadata_path}"
            )

        try:
            with self.metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid model metadata JSON: {self.metadata_path}"
            ) from exc

        if not isinstance(metadata, dict):
            raise ValueError(
                "Model metadata must contain a JSON object."
            )

        required_fields = [
            "model_name",
            "model_version",
            "model_type",
            "framework",
            "artifact",
            "status",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in metadata
        ]

        if missing_fields:
            raise ValueError(
                "Model metadata is missing required fields: "
                + ", ".join(missing_fields)
            )

        self._metadata = metadata

        return self._metadata

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata describing the active model.
        """
        return dict(self.load())

    def get_model_version(self) -> str:
        return str(self.load()["model_version"])

    def get_model_name(self) -> str:
        return str(self.load()["model_name"])