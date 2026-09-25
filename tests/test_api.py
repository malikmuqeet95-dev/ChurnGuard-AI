from fastapi.testclient import TestClient
from plotly import data

from backend.app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200


# def test_health_endpoint():
#     response = client.get("/health")

#     assert response.status_code == 200

#     data = response.json()

#     assert data["status"] == "healthy"
#     assert data["prediction_service_loaded"] is True


# def test_ready_endpoint():
#     response = client.get("/ready")

#     assert response.status_code == 200

#     data = response.json()

#     assert data["status"] == "ready"


def test_prediction_endpoint():
    payload = {
        "tenure": 12,
        "forecast_horizon": 6,
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"

    assert "hazard_ratio" in data
    assert "projected_churn" in data
    assert "projected_retention" in data
    assert "risk_tier" in data
    assert "recommended_action" in data
    assert "forecast_curve" in data
    assert "explanation" in data
    assert "business_decision" in data

    decision = data[
    "business_decision"
]

    assert "priority" in decision
    assert "strategy" in decision
    assert "recommended_actions" in decision

    assert "action_priority" in data

    priority = data[
        "action_priority"
    ]

    assert "priority" in priority
    assert "action_window" in priority
    assert "workflow" in priority
    assert "recommended_actions" in priority


def test_invalid_monthly_charges():
    payload = {
        "tenure": 12,
        "forecast_horizon": 6,
        "MonthlyCharges": 1000,
        "TotalCharges": 900,
    }

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_unknown_feature_is_rejected():
    payload = {
        "tenure": 12,
        "forecast_horizon": 6,
        "MonthlyCharges": 75,
        "TotalCharges": 900,
        "unknown_feature": 123,
    }

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_what_if_endpoint():

    payload = {
        "customer": {
            "tenure": 3,
            "forecast_horizon": 6,

            "MonthlyCharges": 75.0,
            "TotalCharges": 225.0,

            "SeniorCitizen": 0,

            "gender": "Male",

            "Partner": "No",
            "Dependents": "No",

            "PhoneService": "Yes",
            "MultipleLines": "No",

            "InternetService":
                "Fiber optic",

            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",

            "StreamingTV": "No",
            "StreamingMovies": "No",

            "Contract":
                "Month-to-month",

            "PaperlessBilling": "Yes",

            "PaymentMethod":
                "Electronic check",
        },

        "scenarios": [
            {
                "name": "Annual Contract",
                "changes": {
                    "Contract": "One year",
                },
            }
        ],
    }

    response = client.post(
        "/api/what-if",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"

    assert "baseline" in data

    assert "scenarios" in data

    assert data["scenario_count"] == 1

    assert (
        data["scenarios"][0]["rank"]
        == 1
    )


def test_what_if_invalid_field():

    payload = {
        "customer": {
            "tenure": 3,
            "forecast_horizon": 6,

            "MonthlyCharges": 75.0,
            "TotalCharges": 225.0,

            "SeniorCitizen": 0,

            "gender": "Male",

            "Partner": "No",
            "Dependents": "No",

            "PhoneService": "Yes",
            "MultipleLines": "No",

            "InternetService":
                "Fiber optic",

            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",

            "StreamingTV": "No",
            "StreamingMovies": "No",

            "Contract":
                "Month-to-month",

            "PaperlessBilling": "Yes",

            "PaymentMethod":
                "Electronic check",
        },

        "scenarios": [
            {
                "name": "Invalid",
                "changes": {
                    "tenure": 20,
                },
            }
        ],
    }

    response = client.post(
        "/api/what-if",
        json=payload,
    )

    assert response.status_code == 400


def test_prediction_endpoint_contains_retention_decision():
    payload = {
        "tenure": 3,
        "forecast_horizon": 6,
        "MonthlyCharges": 70.0,
        "TotalCharges": 210.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        "retention_decision"
        in data
    )

    decision = data[
        "retention_decision"
    ]

    assert (
        decision["decision_status"]
        == "READY"
    )


def test_prediction_endpoint_retention_decision_has_recommendation():
    payload = {
        "tenure": 3,
        "forecast_horizon": 6,
        "MonthlyCharges": 70.0,
        "TotalCharges": 210.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 200

    decision = response.json()[
        "retention_decision"
    ]

    assert (
        decision["recommendation"]
        in {
            "RECOMMEND_INTERVENTION",
            "CONSIDER_INTERVENTION",
            "REVIEW_ECONOMIC_PRIORITY",
            "HUMAN_REVIEW_REQUIRED",
            "MONITOR_AND_REVIEW",
            "NO_TARGETED_INTERVENTION",
        }
    )


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"


def test_ready_endpoint_contains_readiness_checks():
    response = client.get("/ready")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert body["environment"]
    assert "checks" in body

    assert body["checks"]["model"] is True
    assert body["checks"]["scaler"] is True
    assert body["checks"]["features"] is True
    assert body["checks"]["prediction_engine"] is True

def test_model_metadata_endpoint():
    response = client.get("/api/model")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert "model" in body

    model = body["model"]

    assert model["model_name"]
    assert model["model_version"]
    assert model["model_type"]
    assert model["framework"]
    assert model["artifact"]
    assert model["status"]


def test_prediction_contains_model_metadata():
    payload = {
        "tenure": 12,
        "forecast_horizon": 6,
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["model_name"]
    assert body["model_version"]