
from __future__ import annotations

import json

from backend.src.model_registry import (
    ModelArtifactMetadata,
    ModelArtifactRegistry,
)


def test_sha256_is_deterministic(tmp_path):
    artifact_path = tmp_path / "model.pkl"
    artifact_path.write_bytes(b"test-model-content")

    first_hash = (
        ModelArtifactRegistry.calculate_sha256(
            artifact_path
        )
    )

    second_hash = (
        ModelArtifactRegistry.calculate_sha256(
            artifact_path
        )
    )

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_model_version_contains_model_name_and_hash():
    version = (
        ModelArtifactRegistry.create_model_version(
            model_name="cox_ph",
            artifact_sha256="a" * 64,
        )
    )

    assert version.startswith("cox_ph_")
    assert version.endswith("aaaaaaaa")


def test_artifact_registration_creates_version_directory(
    tmp_path,
):
    source_path = tmp_path / "source_model.pkl"
    source_path.write_bytes(b"model-content")

    registry = ModelArtifactRegistry(
        tmp_path / "registry"
    )

    artifact_hash = (
        registry.calculate_sha256(source_path)
    )

    version = (
        registry.create_model_version(
            model_name="cox_ph",
            artifact_sha256=artifact_hash,
        )
    )

    metadata = ModelArtifactMetadata(
        model_name="cox_ph",
        model_version=version,
        model_type="CoxPHFitter",
        artifact_file="model.pkl",
        artifact_sha256=artifact_hash,
        created_at_utc="2026-09-20T00:00:00+00:00",
        penalizer=0.05,
        random_state=42,
        validation_fraction=0.20,
        training_rows=100,
        validation_rows=25,
        training_c_index=0.90,
        validation_c_index=0.85,
        feature_columns=("feature_a", "feature_b"),
    )

    version_dir = registry.register_artifact(
        artifact_path=source_path,
        metadata=metadata,
        evaluation={
            "validation_c_index": 0.85,
        },
    )

    assert version_dir.exists()
    assert (version_dir / "model.pkl").exists()
    assert (version_dir / "metadata.json").exists()
    assert (version_dir / "evaluation.json").exists()

    with (
        version_dir / "metadata.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_metadata = json.load(file)

    assert saved_metadata["model_name"] == "cox_ph"
    assert saved_metadata["model_version"] == version
    assert saved_metadata["validation_c_index"] == 0.85 or (
        saved_metadata["validation_c_index"] == 0.85
    )

    # Corrected evaluation check: read from evaluation.json
    evaluation_path = version_dir / "evaluation.json"
    with evaluation_path.open("r", encoding="utf-8") as file:
        saved_evaluation = json.load(file)

    assert saved_evaluation["validation_c_index"] == 0.85

def test_manifest_can_be_updated_and_loaded(tmp_path):
    registry = ModelArtifactRegistry(
        tmp_path / "registry"
    )

    manifest_path = registry.update_manifest(
        model_name="cox_ph",
        model_version="cox_ph_test_version",
        metadata_path="versions/cox_ph/test/metadata.json",
    )

    assert manifest_path.exists()

    manifest = registry.load_manifest()

    assert (
        manifest["models"]["cox_ph"][
            "latest_registered_version"
        ]
        == "cox_ph_test_version"
    )