
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.src.drift_monitoring import (
    generate_drift_report,
)


def test_identical_data_has_low_drift():
    reference = pd.DataFrame(
        {
            "MonthlyCharges": np.arange(
                1,
                101,
                dtype=float,
            ),
        }
    )

    current = reference.copy()

    report = generate_drift_report(
        reference_df=reference,
        current_df=current,
    )

    assert len(report.results) == 1
    assert (
        report.results[0].drift_level
        == "LOW"
    )
    assert report.drifted_features == []


def test_shifted_data_can_have_significant_drift():
    reference = pd.DataFrame(
        {
            "MonthlyCharges": np.arange(
                1,
                101,
                dtype=float,
            ),
        }
    )

    current = pd.DataFrame(
        {
            "MonthlyCharges": np.arange(
                1001,
                1101,
                dtype=float,
            ),
        }
    )

    report = generate_drift_report(
        reference_df=reference,
        current_df=current,
    )

    assert len(report.results) == 1
    assert (
        report.results[0].psi >= 0.25
    )
    assert (
        report.results[0].drift_level
        == "SIGNIFICANT"
    )


def test_missing_values_are_reported():
    reference = pd.DataFrame(
        {
            "MonthlyCharges": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
        }
    )

    current = pd.DataFrame(
        {
            "MonthlyCharges": [
                10.0,
                None,
                30.0,
                None,
            ],
        }
    )

    report = generate_drift_report(
        reference_df=reference,
        current_df=current,
    )

    result = report.results[0]

    assert (
        result.reference_missing_rate
        == 0.0
    )

    assert (
        result.current_missing_rate
        == 0.5
    )


def test_only_shared_numeric_features_are_evaluated():
    reference = pd.DataFrame(
        {
            "MonthlyCharges": [10.0, 20.0],
            "Contract": ["Monthly", "Annual"],
        }
    )

    current = pd.DataFrame(
        {
            "MonthlyCharges": [10.0, 20.0],
            "NewFeature": [1, 2],
        }
    )

    report = generate_drift_report(
        reference_df=reference,
        current_df=current,
    )

    names = {
        result.feature_name
        for result in report.results
    }

    assert names == {
        "MonthlyCharges"
    }