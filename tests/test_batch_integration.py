import pandas as pd

from backend.src.batch_predictor import predict_batch
from tests.test_batch_validator import make_valid_customer


def build_large_customer_dataframe(count=7043):
    rows = []

    for index in range(1, count + 1):
        customer = make_valid_customer(
            f"DEMO-{index:05d}"
        )

        # Create some variation so this is not simply the exact
        # same customer repeated thousands of times.
        customer["tenure"] = (
            ((index - 1) % 72) + 1
        )

        customer["forecast_horizon"] = 6

        customer["MonthlyCharges"] = (
            30.0 + ((index * 7) % 170)
        )

        customer["TotalCharges"] = round(
            customer["MonthlyCharges"]
            * customer["tenure"]
            * 0.95,
            2,
        )

        customer["SeniorCitizen"] = (
            index % 5 == 0
        )

        customer["Contract"] = (
            "Month-to-month"
            if index % 3 == 0
            else "One year"
            if index % 3 == 1
            else "Two year"
        )

        rows.append(customer)

    return pd.DataFrame(rows)


def test_real_batch_prediction_7043_customers():
    dataframe = build_large_customer_dataframe(
        7043
    )

    report = predict_batch(
        dataframe
    )

    assert len(report) == 7043

    expected_columns = [
        "Customer ID",
        "Risk Tier",
        "Hazard Ratio",
        "Churn Probability",
        "Retention Probability",
        "Revenue at Risk",
        "Risk Window",
        "Recommended Action",
        "Priority",
        "Priority Score",
    ]

    assert list(report.columns) == expected_columns

    assert report["Customer ID"].nunique() == 7043

    assert report["Hazard Ratio"].notna().all()

    assert report["Churn Probability"].between(
        0,
        1,
    ).all()

    assert report["Retention Probability"].between(
        0,
        1,
    ).all()

    assert report["Revenue at Risk"].ge(
        0
    ).all()

    assert report["Risk Tier"].notna().all()

    assert report["Recommended Action"].notna().all()

    assert report["Priority"].notna().all()

    assert report["Priority Score"].notna().all()