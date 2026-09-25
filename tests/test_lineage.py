
from __future__ import annotations

import json

from backend.src.lineage import (
    create_training_lineage,
)


def test_training_lineage_contains_dataset_and_model_metadata(
    tmp_path,
):
    dataset_path = (
        tmp_path / "dataset.csv"
    )

    dataset_path.write_text(
        "tenure_months,churn_event\n"
        "12,0\n"
        "24,1\n",
        encoding="utf-8",
    )

    lineage = create_training_lineage(
        dataset_path=dataset_path,
        model_name="cox_ph",
        model_version="cox_ph_test_version",
        feature_pipeline_version="1.0.0",
        training_run_id="test-run-123",
    )

    result = lineage.to_dict()

    assert (
        result["dataset_name"]
        == "telco_churn_features"
    )

    assert result["dataset_version"]
    assert result["dataset_path"]
    assert (
        result["feature_pipeline_version"]
        == "1.0.0"
    )

    assert result["model_name"] == "cox_ph"
    assert (
        result["model_version"]
        == "cox_ph_test_version"
    )

    assert (
        result["training_run_id"]
        == "test-run-123"
    )


def test_training_lineage_can_be_saved(
    tmp_path,
):
    dataset_path = (
        tmp_path / "dataset.csv"
    )

    dataset_path.write_text(
        "tenure_months,churn_event\n"
        "12,0\n",
        encoding="utf-8",
    )

    lineage = create_training_lineage(
        dataset_path=dataset_path,
        model_name="cox_ph",
        model_version="cox_ph_test_version",
    )

    output_path = (
        tmp_path / "lineage.json"
    )

    saved_path = lineage.save_json(
        output_path
    )

    assert saved_path.exists()

    with saved_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_data = json.load(file)

    assert (
        saved_data["model_name"]
        == "cox_ph"
    )

    assert (
        saved_data["model_version"]
        == "cox_ph_test_version"
    )

    assert saved_data["dataset_version"]