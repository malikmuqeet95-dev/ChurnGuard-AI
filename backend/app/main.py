from pathlib import Path
import logging
import time

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import status
from fastapi.responses import JSONResponse

from backend.app.operational_metrics import operational_metrics
from backend.app.api_key_security import verify_api_key
from backend.app.security_middleware import (
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware,
)
from backend.app.model_metadata import ModelMetadataService
from backend.app.health_service import HealthService
from backend.app.request_middleware import RequestIDMiddleware
from backend.app.request_context import get_request_id
from backend.app.error_handlers import register_exception_handlers
from backend.app.config import settings
from backend.app.logging_config import configure_logging, get_logger
from backend.app.prediction_service import PredictionService
from backend.app.schemas import (
    CustomerPayload,
    PredictionResponse,
    WhatIfRequest,
    WhatIfResponse,
)


# ============================================================
# LOGGING
# ============================================================

configure_logging(settings.log_level)
logger = get_logger(__name__)

# logging.basicConfig(
#     level=logging.INFO,
#     format=(
#         "%(asctime)s - "
#         "%(levelname)s - "
#         "%(name)s - "
#         "%(message)s"
#     ),
# )

# logger = logging.getLogger("churnguard.api")


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"
CSS_DIR = FRONTEND_DIR / "css"
JS_DIR = FRONTEND_DIR / "js"


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title=settings.app_name,
    description=(
        "Production-oriented customer churn survival "
        "and retention decision intelligence API."
    ),
    version=settings.app_version,
    docs_url="/docs" if settings.enable_api_docs else None,
    redoc_url="/redoc" if settings.enable_api_docs else None,
    openapi_url="/openapi.json"
    if settings.enable_api_docs
    else None,
)

register_exception_handlers(app)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(RequestIDMiddleware)

logger.info(
    "Starting %s version %s in %s environment",
    settings.app_name,
    settings.app_version,
    settings.app_environment,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-request-id", "X-Request-ID"],
)


# ============================================================
# STATIC FRONTEND
# ============================================================

if CSS_DIR.exists():
    app.mount(
        "/css",
        StaticFiles(
            directory=str(CSS_DIR)
        ),
        name="css",
    )

if JS_DIR.exists():
    app.mount(
        "/js",
        StaticFiles(
            directory=str(JS_DIR)
        ),
        name="js",
    )


# ============================================================
# PREDICTION SERVICE
# ============================================================

try:
    prediction_service = PredictionService()
    health_service = HealthService(prediction_service)
    model_metadata_service = ModelMetadataService()

    logger.info(
        "Prediction & Health & Model Metadata services initialized successfully."
    )

except Exception as exc:
    prediction_service = None
    health_service = None
    model_metadata_service = ModelMetadataService()

    logger.exception(
        "Prediction & Health & Model Metadata services initialization failed: %s",
        exc,
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Frontend index.html not found: "
                f"{index_file}"
            ),
        )

    return FileResponse(
        str(index_file)
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    result = health_service.liveness()

    logger.debug(
        "Health check completed | status=%s",
        result["status"],
    )

    return result

# ============================================================
# READINESS CHECK
# ============================================================

@app.get("/ready")
def ready():
    result = health_service.readiness()

    logger.debug(
        "Readiness check completed | status=%s | checks=%s",
        result["status"],
        result["checks"],
    )

    if result["status"] != "ready":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=result,
        )

    return result

# ============================================================
# OPERATIONAL METRICS
# ============================================================

@app.get("/metrics")
def get_metrics():
    """Return non-sensitive application operational metrics."""

    return {
        "status": "ok",
        "environment": settings.app_environment,
        "metrics": operational_metrics.snapshot(),
    }

# ============================================================
# MODEL METADATA
# ============================================================

@app.get("/api/model")
def model_metadata():
    metadata = model_metadata_service.get_metadata()

    logger.debug(
        "Model metadata requested | model=%s | version=%s",
        metadata.get("model_name"),
        metadata.get("model_version"),
    )

    return {
        "status": "ok",
        "model": metadata,
    }



# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
)
def predict_churn(
    payload: CustomerPayload,
    _: None = Depends(verify_api_key),
):
    
    operational_metrics.record_prediction()
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service is unavailable."
            ),
        )

    start_time = time.perf_counter()

    try:
        customer_data = payload.model_dump()

        result = prediction_service.predict(
            customer_data
        )
        operational_metrics.record_prediction_result(
            success=True,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "Prediction completed successfully | request_id=%s | latency_ms=%.2f | risk_tier=%s | projected_churn=%.2f | target_month=%s",
            get_request_id(),
            elapsed_ms,
            result.get("risk_tier"),
            result.get("projected_churn"),
            result.get("target_month"),
        )

        return result

    except ValueError as exc:
        operational_metrics.record_prediction_result(success=False)
        logger.warning(
            "Prediction validation error: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        operational_metrics.record_prediction_result(
            success=False,
        )

        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=500,
            detail="Internal prediction service error.",
        )


@app.post(
    "/api/what-if",
    response_model=WhatIfResponse,
)
def simulate_interventions(
    payload: WhatIfRequest,
    _: None = Depends(verify_api_key),
):
    operational_metrics.record_what_if()
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service is unavailable."
            ),
        )

    start_time = time.perf_counter()

    try:

        customer_data = (
            payload.customer.model_dump()
        )

        scenarios = [
            scenario.model_dump()
            for scenario
            in payload.scenarios
        ]

        result = (
            prediction_service
            .simulate_interventions(
                customer_data=customer_data,
                scenarios=scenarios,
            )
        )
        operational_metrics.record_what_if_result(
            success=True,
        )
        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.info(
            "What-if simulation completed "
            "| request_id=%s "
            "| scenarios=%s "
            "| best=%s "
            "| latency=%.2fms",
            get_request_id(),
            result["scenario_count"],
            result["best_scenario"],
            elapsed_ms,
        )

        return result
    
    except ValueError as exc:
        operational_metrics.record_what_if_result(
            success=False,
        )
        logger.warning(
            "What-if validation error: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        operational_metrics.record_what_if_result(
            success=False,
        )
        logger.exception("Unexpected error during what-if simulation")

        raise HTTPException(
            status_code=500,
            detail="Internal simulation service error.",
        )
    