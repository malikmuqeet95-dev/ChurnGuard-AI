from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
#
# customer_churn_survival_mlops/
#
# config.py is located at:
# customer_churn_survival_mlops/backend/app/config.py
#
# parents[0] -> backend/app
# parents[1] -> backend
# parents[2] -> project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_DIR = PROJECT_ROOT / "backend"
SRC_DIR = BACKEND_DIR / "src"

DEFAULT_MODEL_PATH = (
    SRC_DIR / "models" / "cox_ph_model.pkl"
)
DEFAULT_MODEL_METADATA_PATH = (
    SRC_DIR / "models" / "model_metadata.json"
)

DEFAULT_SCALER_PATH = (
    SRC_DIR / "data" / "processed" / "scaler.pkl"
)

DEFAULT_FEATURES_PATH = (
    SRC_DIR / "data" / "processed" / "telco_churn_features.csv"
)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

class Settings(BaseSettings):
    """
    Centralized application configuration.

    Values can come from:
        1. Environment variables
        2. .env file
        3. Defaults defined below

    Environment variables take priority over .env values,
    and .env values take priority over defaults.
    """

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    app_name: str = Field(
        default="ChurnGuard AI",
        description="Application name",
    )

    app_version: str = Field(
        default="2.1.0",
        description="Application version",
    )

    app_environment: str = Field(
        default="development",
        validation_alias=AliasChoices("APP_ENVIRONMENT", "APP_ENV"),
        description="Application environment",
    )

    # --------------------------------------------------------
    # API Server
    # --------------------------------------------------------

    api_host: str = Field(
        default="127.0.0.1",
        description="API server host",
    )

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port",
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log_level: str = Field(
        default="INFO",
        description="Application logging level",
    )

    # --------------------------------------------------------
    # CORS
    # --------------------------------------------------------

    cors_origins: str = Field(
        default="http://localhost:5173",
        description=(
            "Comma-separated list of allowed CORS origins. "
            "Use '*' to allow all origins during development."
        ),
    )

    # --------------------------------------------------------
    # ML Model Artifacts
    # --------------------------------------------------------

    model_path: Path = Field(
        default=DEFAULT_MODEL_PATH,
        description="Path to the trained Cox proportional hazards model",
    )

    scaler_path: Path = Field(
        default=DEFAULT_SCALER_PATH,
        description="Path to the persisted feature scaler",
    )

    features_path: Path = Field(
        default=DEFAULT_FEATURES_PATH,
        description="Path to processed feature dataset",
    )

    model_metadata_path: Path = Field(
        default=DEFAULT_MODEL_METADATA_PATH
    )
    # --------------------------------------------------------
    # Business Configuration
    # --------------------------------------------------------

    customer_value: float = Field(
        default=1200.0,
        ge=0,
        description=(
            "Default estimated customer value used by the "
            "intervention ROI engine."
        ),
    )

    # --------------------------------------------------------
    # API SECURITY
    # --------------------------------------------------------

    require_api_key: bool = Field(
        default=False,
        description=(
            "Require an API key for protected API endpoints."
        ),
    )

    api_key: str = Field(
        default="",
        description=(
            "API key used to authenticate protected API requests."
        ),
    )
    
    enable_api_docs: bool = Field(default=True)
    
    max_request_body_bytes: int = Field(
        default=1_048_576,
        ge=1024,
        le=10_485_760,
    )

    # --------------------------------------------------------
    # Pydantic Settings Configuration
    # --------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================
    # HELPER PROPERTIES
    # ========================================================

    @property
    def cors_origins_list(self) -> List[str]:
        """
        Convert comma-separated CORS origins into a list.

        Example:

            CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

        becomes:

            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        """

        origins = [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

        return origins

    @property
    def is_development(self) -> bool:
        """Return True when running in development mode."""

        return self.app_environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Return True when running in production mode."""

        return self.app_environment.lower() == "production"


# ============================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================

settings = Settings()