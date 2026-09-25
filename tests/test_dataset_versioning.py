"""
Phase 7 - Step 3
Dataset Versioning Tests
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.dataset_versioning import DatasetVersionManager


def create_test_csv(tmp_path):

    dataframe = pd.DataFrame(
        {
            "tenure_months": [1, 12, 24],
            "churn_event": [0, 1, 0],
            "feature_a": [1.0, 2.0, 3.0],
        }
    )

    path = tmp_path / "dataset.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    return path


def test_sha256_is_deterministic(tmp_path):

    path = create_test_csv(tmp_path)

    manager = DatasetVersionManager()

    first_hash = manager.calculate_sha256(path)
    second_hash = manager.calculate_sha256(path)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_dataframe_fingerprint_is_deterministic():

    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )

    manager = DatasetVersionManager()

    first_hash = (
        manager.calculate_dataframe_fingerprint(
            dataframe
        )
    )

    second_hash = (
        manager.calculate_dataframe_fingerprint(
            dataframe
        )
    )

    assert first_hash == second_hash


def test_different_dataframe_has_different_fingerprint():

    first_dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3],
        }
    )

    second_dataframe = pd.DataFrame(
        {
            "a": [1, 2, 4],
        }
    )

    manager = DatasetVersionManager()

    first_hash = (
        manager.calculate_dataframe_fingerprint(
            first_dataframe
        )
    )

    second_hash = (
        manager.calculate_dataframe_fingerprint(
            second_dataframe
        )
    )

    assert first_hash != second_hash


def test_manifest_contains_required_metadata(tmp_path):

    path = create_test_csv(tmp_path)

    manager = DatasetVersionManager()

    manifest = manager.create_manifest(path)

    required_fields = [
        "dataset_name",
        "dataset_version",
        "schema_version",
        "feature_pipeline_version",
        "source",
        "file_name",
        "file_sha256",
        "dataframe_sha256",
        "row_count",
        "column_count",
        "columns",
        "created_at_utc",
    ]

    for field in required_fields:
        assert field in manifest

    assert manifest["row_count"] == 3
    assert manifest["column_count"] == 3


def test_manifest_can_be_saved_and_loaded(tmp_path):

    path = create_test_csv(tmp_path)

    manager = DatasetVersionManager()

    manifest = manager.create_manifest(path)

    manifest_path = tmp_path / "manifest.json"

    saved_path = manager.save_manifest(
        manifest,
        manifest_path,
    )

    assert saved_path.is_file()

    loaded_manifest = manager.load_manifest(
        manifest_path
    )

    assert loaded_manifest == manifest


def test_manifest_verification_succeeds_for_unchanged_file(
    tmp_path,
):

    path = create_test_csv(tmp_path)

    manager = DatasetVersionManager()

    manifest = manager.create_manifest(path)

    assert manager.verify_manifest(
        path,
        manifest,
    ) is True


def test_manifest_verification_fails_after_data_change(
    tmp_path,
):

    path = create_test_csv(tmp_path)

    manager = DatasetVersionManager()

    manifest = manager.create_manifest(path)

    dataframe = pd.read_csv(path)

    dataframe.loc[0, "feature_a"] = 999.0

    dataframe.to_csv(
        path,
        index=False,
    )

    assert manager.verify_manifest(
        path,
        manifest,
    ) is False


def test_missing_dataset_raises_error(tmp_path):

    manager = DatasetVersionManager()

    missing_path = (
        tmp_path / "does_not_exist.csv"
    )

    with pytest.raises(FileNotFoundError):

        manager.calculate_sha256(
            missing_path
        )


def test_real_dataset_manifest_can_be_created():

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "backend"
        / "src"
        / "data"
        / "processed"
        / "telco_churn_features.csv"
    )

    manager = DatasetVersionManager()

    manifest = manager.create_manifest(
        dataset_path
    )

    assert manifest["row_count"] > 0
    assert manifest["column_count"] > 0
    assert len(manifest["file_sha256"]) == 64
    assert len(manifest["dataframe_sha256"]) == 64