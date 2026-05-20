# src/models/schemas.py

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Input schema — validates one customer record before it reaches the model."""

    model_config = ConfigDict(populate_by_name=True,
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20,
            }
        },
    )

    # ── Numeric features ──────────────────────────────────────────────────────
    tenure: int = Field(
        ..., ge=0, alias="tenure",
        description="Months as a customer",
        examples=[12],
    )
    monthly_charges: float = Field(
        ..., gt=0, alias="MonthlyCharges",
        description="Monthly bill in USD",
        examples=[70.35],
    )
    total_charges: float = Field(
        ..., ge=0, alias="TotalCharges",
        description="Total billed to date in USD",
        examples=[844.20],
    )
    senior_citizen: int = Field(
        ..., ge=0, le=1, alias="SeniorCitizen",
        description="1 if senior citizen, 0 otherwise",
        examples=[0],
    )

    # ── Categorical features ──────────────────────────────────────────────────
    gender: Literal["Male", "Female"] = Field(
        ..., alias="gender",
        description="Customer gender",
        examples=["Female"],
    )
    partner: Literal["Yes", "No"] = Field(
        ..., alias="Partner",
        description="Whether the customer has a partner",
        examples=["Yes"],
    )
    dependents: Literal["Yes", "No"] = Field(
        ..., alias="Dependents",
        description="Whether the customer has dependents",
        examples=["No"],
    )
    phone_service: Literal["Yes", "No"] = Field(
        ..., alias="PhoneService",
        description="Whether the customer has phone service",
        examples=["Yes"],
    )
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        ..., alias="MultipleLines",
        description="Whether the customer has multiple lines",
        examples=["No"],
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., alias="InternetService",
        description="Internet service provider type",
        examples=["Fiber optic"],
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="OnlineSecurity",
        description="Whether the customer has online security",
        examples=["No"],
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="OnlineBackup",
        description="Whether the customer has online backup",
        examples=["Yes"],
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="DeviceProtection",
        description="Whether the customer has device protection",
        examples=["No"],
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="TechSupport",
        description="Whether the customer has tech support",
        examples=["No"],
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="StreamingTV",
        description="Whether the customer streams TV",
        examples=["Yes"],
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="StreamingMovies",
        description="Whether the customer streams movies",
        examples=["Yes"],
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., alias="Contract",
        description="Contract term",
        examples=["Month-to-month"],
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        ..., alias="PaperlessBilling",
        description="Whether the customer uses paperless billing",
        examples=["Yes"],
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ..., alias="PaymentMethod",
        description="Payment method",
        examples=["Electronic check"],
    )

class PredictionResult(BaseModel):
    """Output schema — documents and validates what your pipeline returns."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "churn_prediction": True,
                "churn_probability": 0.7823,
                "customer_id": None,
                "model_version": None,
            }
        }
    )

    churn_prediction: bool = Field(
        ..., description="True if customer is predicted to churn"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of churn between 0 and 1",
        examples=[0.7823],
    )
    customer_id: Optional[str] = Field(
        None, description="Echo of customer ID if provided in request"
    )
    model_version: Optional[str] = Field(
        None, description="Version of the model that made this prediction"
    )

class ErrorResponse(BaseModel):
    """Consistent error shape returned by all error handlers."""

    error: str = Field(..., description="Snake_case error type identifier")
    detail: str = Field(..., description="Human-readable explanation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "prediction_error",
                "detail": "Model not loaded. Call load() first.",
            }
        }
    )