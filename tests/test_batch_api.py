import io

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from tests.test_batch_validator import make_valid_customer


client = TestClient(app)


def make_csv_bytes(rows=2):
    dataframe = pd.DataFrame(
        [
            make_valid_customer(
                f"CUST-{index:03d}"
            )
            for index in range(1, rows + 1)
        ]
    )

    buffer = io.StringIO()

    dataframe.to_csv(
        buffer,
        index=False,
    )

    return buffer.getvalue().encode("utf-8")


def test_batch_predict_endpoint_returns_csv():
    response = client.post(
        "/api/batch-predict",
        files={
            "file": (
                "customers.csv",
                make_csv_bytes(2),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    assert (
        response.headers["content-type"]
        .startswith("text/csv")
    )

    assert (
        "customer_risk_report.csv"
        in response.headers["content-disposition"]
    )

    report = pd.read_csv(
        io.StringIO(response.text)
    )

    assert len(report) == 2

    assert "Customer ID" in report.columns
    assert "Risk Tier" in report.columns
    assert "Hazard Ratio" in report.columns
    assert "Churn Probability" in report.columns
    assert "Retention Probability" in report.columns
    assert "Revenue at Risk" in report.columns
    assert "Risk Window" in report.columns
    assert "Recommended Action" in report.columns


def test_batch_predict_rejects_non_csv():
    response = client.post(
        "/api/batch-predict",
        files={
            "file": (
                "customers.txt",
                b"not a csv",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert "CSV" in response.json()["detail"]


def test_batch_predict_rejects_empty_file():
    response = client.post(
        "/api/batch-predict",
        files={
            "file": (
                "customers.csv",
                b"",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400


def test_batch_predict_rejects_invalid_customer_data():
    customer = make_valid_customer("BAD-001")
    customer["Contract"] = "INVALID"

    dataframe = pd.DataFrame([customer])

    buffer = io.StringIO()
    dataframe.to_csv(
        buffer,
        index=False,
    )

    response = client.post(
        "/api/batch-predict",
        files={
            "file": (
                "customers.csv",
                buffer.getvalue().encode("utf-8"),
                "text/csv",
            )
        },
    )

    assert response.status_code == 422