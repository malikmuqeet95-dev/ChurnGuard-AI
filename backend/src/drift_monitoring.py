
"""
Lightweight numerical data drift monitoring.

This module compares a reference dataset with a current dataset.
It does not trigger retraining or modify production artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift result for one feature."""

    feature_name: str
    psi: float
    drift_level: str
    reference_missing_rate: float
    current_missing_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary."""

        return {
            "feature_name": self.feature_name,
            "psi": self.psi,
            "drift_level": self.drift_level,
            "reference_missing_rate": (
                self.reference_missing_rate
            ),
            "current_missing_rate": (
                self.current_missing_rate
            ),
        }


@dataclass(frozen=True)
class DriftReport:
    """Complete drift report."""

    results: tuple[FeatureDriftResult, ...]

    @property
    def drifted_features(self) -> list[str]:
        """Return features with significant drift."""

        return [
            result.feature_name
            for result in self.results
            if result.drift_level == "SIGNIFICANT"
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary."""

        return {
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "drifted_features": self.drifted_features,
        }


def _calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Calculate PSI using reference-derived quantile bins.

    Constant or insufficiently variable features are handled safely.
    """

    reference = reference[
        np.isfinite(reference)
    ]

    current = current[
        np.isfinite(current)
    ]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    edges = np.quantile(
        reference,
        quantiles,
    )

    edges = np.unique(edges)

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=edges,
    )

    current_counts, _ = np.histogram(
        current,
        bins=edges,
    )

    reference_proportions = (
        reference_counts / len(reference)
    )

    current_proportions = (
        current_counts / len(current)
    )

    epsilon = 1e-6

    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )

    current_proportions = np.clip(
        current_proportions,
        epsilon,
        None,
    )

    psi = np.sum(
        (
            current_proportions
            - reference_proportions
        )
        * np.log(
            current_proportions
            / reference_proportions
        )
    )

    return float(psi)


def _classify_drift(
    psi: float,
) -> str:
    """Classify PSI using configurable project heuristics."""

    if psi < 0.10:
        return "LOW"

    if psi < 0.25:
        return "MODERATE"

    return "SIGNIFICANT"


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    bins: int = 10,
) -> DriftReport:
    """
    Generate a numerical feature drift report.

    Only numeric features shared by both datasets are evaluated.
    """

    if feature_columns is None:
        feature_columns = [
            column
            for column in reference_df.columns
            if column in current_df.columns
            and pd.api.types.is_numeric_dtype(
                reference_df[column]
            )
            and pd.api.types.is_numeric_dtype(
                current_df[column]
            )
        ]

    results: list[FeatureDriftResult] = []

    for feature_name in feature_columns:
        if (
            feature_name not in reference_df.columns
            or feature_name not in current_df.columns
        ):
            continue

        reference_series = reference_df[
            feature_name
        ]

        current_series = current_df[
            feature_name
        ]

        reference_missing_rate = float(
            reference_series.isna().mean()
        )

        current_missing_rate = float(
            current_series.isna().mean()
        )

        reference_values = (
            pd.to_numeric(
                reference_series,
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        current_values = (
            pd.to_numeric(
                current_series,
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        psi = _calculate_psi(
            reference=reference_values,
            current=current_values,
            bins=bins,
        )

        results.append(
            FeatureDriftResult(
                feature_name=feature_name,
                psi=psi,
                drift_level=_classify_drift(psi),
                reference_missing_rate=(
                    reference_missing_rate
                ),
                current_missing_rate=(
                    current_missing_rate
                ),
            )
        )

    return DriftReport(
        results=tuple(results)
    )