from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.config import settings


class HealthService:
    """
    Provides application health and readiness checks.

    /health answers:
        Is the application process alive?

    /ready answers:
        Is the application capable of serving predictions?
    """

    def __init__(self, prediction_service: Any | None = None) -> None:
        self.prediction_service = prediction_service

    def liveness(self) -> dict[str, Any]:
        """
        Lightweight process-level health check.

        This intentionally does not inspect model files or run predictions.
        """
        return {
            "status": "ok",
        }

    def _file_exists(self, path: Path) -> bool:
        return path.exists() and path.is_file()

    def readiness(self) -> dict[str, Any]:
        """
        Check whether required application resources are available.
        """

        model_ok = self._file_exists(settings.model_path)
        scaler_ok = self._file_exists(settings.scaler_path)
        features_ok = self._file_exists(settings.features_path)

        prediction_engine_ok = self.prediction_service is not None

        checks = {
            "model": model_ok,
            "scaler": scaler_ok,
            "features": features_ok,
            "prediction_engine": prediction_engine_ok,
        }

        ready = all(checks.values())

        return {
            "status": "ready" if ready else "not_ready",
            "environment": settings.app_environment,
            "checks": checks,
        }