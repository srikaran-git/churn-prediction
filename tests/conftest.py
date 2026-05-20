# tests/conftest.py
"""
Shared fixtures for all test modules.
Fixtures here are auto-discovered by pytest — no imports needed in test files.
"""

import pandas as pd
import pytest

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import app, predictor
from src.models.schemas import PredictionResult


@pytest.fixture
def mock_prediction():
    """Known PredictionResult to return from mocked predictor."""
    return PredictionResult(
        churn_prediction=True,
        churn_probability=0.7823,
    )


@pytest.fixture
def client(mock_prediction):
    """
    TestClient with model fully loaded (simulated).
    Use for happy-path and validation tests.
    """
    with patch.object(predictor, "load"), \
         patch.object(predictor, "predict", return_value=mock_prediction):
        with TestClient(app) as test_client:
            predictor.pipeline = MagicMock()  # health check sees loaded model
            yield test_client
    predictor.pipeline = None  # cleanup after test


@pytest.fixture
def client_unloaded():
    """
    TestClient where load() was a no-op — pipeline stays None.
    Use only for testing the 503 health check path.
    """
    with patch.object(predictor, "load"):
        with TestClient(app) as test_client:
            yield test_client

@pytest.fixture
def raw_churn_df():
    """
    Minimal realistic DataFrame that mirrors the churn CSV structure.
    Use this instead of loading the real file — tests must never depend on disk.
    """
    return pd.DataFrame(
        {
            "customerID": ["001", "002", "003", "004"],
            "gender": ["Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [12, 5, 24, 1],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["Yes", "No", "Yes", "No"],
            "OnlineBackup": ["No", "Yes", "No", "No"],
            "DeviceProtection": ["Yes", "No", "Yes", "No"],
            "TechSupport": ["No", "No", "Yes", "No"],
            "StreamingTV": ["No", "Yes", "No", "No"],
            "StreamingMovies": ["No", "Yes", "No", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "TotalCharges": ["358.20", "284.75", " ", "42.30"],
            "Churn": ["No", "Yes", "No", "No"],
        }
    )


@pytest.fixture
def clean_churn_df(raw_churn_df):
    """
    Same DataFrame but with TotalCharges already converted to float.
    Use when testing functions that expect clean numeric data.
    """
    df = raw_churn_df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df
