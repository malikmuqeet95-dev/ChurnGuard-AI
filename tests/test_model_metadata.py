import json
from pathlib import Path

import pytest

from backend.app.model_metadata import ModelMetadataService


def create_metadata_file(tmp_path: Path) -> Path:
    metadata_path = tmp_path / "model_metadata.json"

    metadata = {
        "model_name": "Test Model",
        "model_version": "1.2.3",
        "model_type": "Cox Proportional Hazards",
        "framework": "lifelines",
        "artifact": "test_model.pkl",
        "status": "active",
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    return metadata_path


def test_metadata_is_loaded(tmp_path):
    metadata_path = create_metadata_file(tmp_path)

    service = ModelMetadataService(metadata_path)

    metadata = service.get_metadata()

    assert metadata["model_name"] == "Test Model"
    assert metadata["model_version"] == "1.2.3"


def test_model_version_is_returned(tmp_path):
    metadata_path = create_metadata_file(tmp_path)

    service = ModelMetadataService(metadata_path)

    assert service.get_model_version() == "1.2.3"


def test_model_name_is_returned(tmp_path):
    metadata_path = create_metadata_file(tmp_path)

    service = ModelMetadataService(metadata_path)

    assert service.get_model_name() == "Test Model"


def test_missing_metadata_file_is_rejected(tmp_path):
    metadata_path = tmp_path / "missing.json"

    service = ModelMetadataService(metadata_path)

    with pytest.raises(FileNotFoundError):
        service.get_metadata()


def test_invalid_json_is_rejected(tmp_path):
    metadata_path = tmp_path / "invalid.json"

    metadata_path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    service = ModelMetadataService(metadata_path)

    with pytest.raises(ValueError):
        service.get_metadata()


def test_missing_required_metadata_field_is_rejected(tmp_path):
    metadata_path = tmp_path / "incomplete.json"

    metadata = {
        "model_name": "Test Model",
        "model_version": "1.0.0"
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    service = ModelMetadataService(metadata_path)

    with pytest.raises(ValueError):
        service.get_metadata()