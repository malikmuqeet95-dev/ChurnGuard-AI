from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.operational_metrics import operational_metrics


client = TestClient(app)


def setup_function():
    operational_metrics.reset()


def test_predict_increments_prediction_counter():
    before = operational_metrics.snapshot()["predictions_total"]

    payload = {
        "tenure": 3,
        "forecast_horizon": 6,
        "SeniorCitizen": 0,
        "MonthlyCharges": 70.0,
        "TotalCharges": 210.0,
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

    after = operational_metrics.snapshot()["predictions_total"]

    assert after == before + 1


def test_what_if_increments_what_if_counter():
    before = operational_metrics.snapshot()["what_if_total"]

    customer_data = {
        "tenure": 3,
        "SeniorCitizen": 0,
        "MonthlyCharges": 70.0,
        "TotalCharges": 210.0,
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

    payload = {
        "customer": customer_data,
        "scenarios": [
            {
                "name": "Add Tech Support",
                "changes": {
                    "TechSupport": "Yes"
                },
            }
        ],
    }

    response = client.post(
        "/api/what-if",
        json=payload,
    )

    assert response.status_code == 200

    after = operational_metrics.snapshot()["what_if_total"]

    assert after == before + 1