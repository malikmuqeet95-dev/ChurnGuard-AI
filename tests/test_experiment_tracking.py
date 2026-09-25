from __future__ import annotations

import mlflow

try:
    from backend.src.experiment_tracking import ExperimentTracker
except ImportError:
    from src.experiment_tracking import ExperimentTracker


def test_experiment_tracker_logs_parameters_and_metrics(tmp_path):
    # Standard SQLite tracking URI avoids the deprecated FileStore backend
    db_path = (tmp_path / "mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{db_path}"

    tracker = ExperimentTracker(
        experiment_name="test-churnguard-experiment",
        tracking_uri=tracking_uri,
    )

    with tracker.start_run(run_name="test-run"):
        tracker.log_parameters(
            {
                "penalizer": 0.05,
                "random_state": 42,
            }
        )

        tracker.log_metrics(
            {
                "training_c_index": 0.90,
                "validation_c_index": 0.85,
            }
        )

        tracker.log_tags(
            {
                "model_type": "CoxPHFitter",
                "environment": "test",
            }
        )

        run_id = tracker.get_active_run_id()
        assert run_id is not None

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    assert run.data.params["penalizer"] == "0.05"
    assert run.data.params["random_state"] == "42"
    assert run.data.metrics["training_c_index"] == 0.90
    assert run.data.metrics["validation_c_index"] == 0.85
    assert run.data.tags["model_type"] == "CoxPHFitter"