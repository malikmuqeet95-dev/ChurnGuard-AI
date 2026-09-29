import pandas as pd

from backend.src.batch_schema import REQUIRED_BATCH_COLUMNS
from backend.src.batch_validator import (
    validate_batch_dataframe,
    validate_batch_or_raise,
)


def make_valid_customer(customer_id="TEST-001"):
    return {
        "Customer ID": customer_id,
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


def test_valid_batch_is_accepted():
    dataframe = pd.DataFrame([
        make_valid_customer(),
        make_valid_customer("TEST-002"),
    ])

    result = validate_batch_dataframe(dataframe)

    assert result.valid is True
    assert result.total_rows == 2
    assert result.valid_rows == 2
    assert result.errors == []


def test_missing_required_column_is_rejected():
    customer = make_valid_customer()
    customer.pop("PaymentMethod")

    dataframe = pd.DataFrame([customer])

    result = validate_batch_dataframe(dataframe)

    assert result.valid is False
    assert "PaymentMethod" in result.errors[0]


def test_invalid_categorical_value_is_rejected():
    customer = make_valid_customer()
    customer["Contract"] = "Invalid Contract"

    dataframe = pd.DataFrame([customer])

    result = validate_batch_dataframe(dataframe)

    assert result.valid is False
    assert any(
        "Contract contains invalid values" in error
        for error in result.errors
    )


def test_invalid_tenure_is_rejected():
    customer = make_valid_customer()
    customer["tenure"] = 100

    dataframe = pd.DataFrame([customer])

    result = validate_batch_dataframe(dataframe)

    assert result.valid is False
    assert any(
        "tenure must be between" in error
        for error in result.errors
    )


def test_duplicate_customer_ids_are_rejected():
    dataframe = pd.DataFrame([
        make_valid_customer("DUPLICATE"),
        make_valid_customer("DUPLICATE"),
    ])

    result = validate_batch_dataframe(dataframe)

    assert result.valid is False
    assert any(
        "Duplicate customer IDs" in error
        for error in result.errors
    )


def test_extra_columns_generate_warning():
    customer = make_valid_customer()
    customer["UnusedColumn"] = "ignored"

    dataframe = pd.DataFrame([customer])

    result = validate_batch_dataframe(dataframe)

    assert result.valid is True
    assert result.warnings
    assert any(
        "UnusedColumn" in warning
        for warning in result.warnings
    )


def test_empty_dataframe_is_rejected():
    dataframe = pd.DataFrame(
        columns=REQUIRED_BATCH_COLUMNS
    )

    result = validate_batch_dataframe(dataframe)

    assert result.valid is False
    assert "Batch dataset is empty." in result.errors


def test_validate_or_raise_returns_normalized_dataframe():
    dataframe = pd.DataFrame([
        make_valid_customer()
    ])

    result = validate_batch_or_raise(dataframe)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.loc[0, "Customer ID"] == "TEST-001"