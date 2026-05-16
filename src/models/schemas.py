# src/models/schemas.py

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerFeatures(BaseModel):
    """Input schema — validates one customer record before it reaches the model."""

    model_config = ConfigDict(populate_by_name=True)

    # ── Numeric features ──────────────────────────────────────
    tenure: int = Field(..., ge=0, alias="tenure", description="Months as a customer")
    monthly_charges: float = Field(..., gt=0, alias="MonthlyCharges", description="Monthly bill")
    total_charges: float = Field(..., ge=0, alias="TotalCharges", description="Total billed to date")
    senior_citizen: int = Field(..., ge=0, le=1, alias="SeniorCitizen", description="1 if senior citizen")

    # ── Categorical features ──────────────────────────────────
    gender: str = Field(..., alias="gender", description="Male or Female")
    partner: str = Field(..., alias="Partner", description="Yes or No")
    dependents: str = Field(..., alias="Dependents", description="Yes or No")
    phone_service: str = Field(..., alias="PhoneService", description="Yes or No")
    multiple_lines: str = Field(..., alias="MultipleLines", description="Yes, No, or No phone service")
    internet_service: str = Field(..., alias="InternetService", description="DSL, Fiber optic, or No")
    online_security: str = Field(..., alias="OnlineSecurity", description="Yes, No, or No internet service")
    online_backup: str = Field(..., alias="OnlineBackup", description="Yes, No, or No internet service")
    device_protection: str = Field(..., alias="DeviceProtection", description="Yes, No, or No internet service")
    tech_support: str = Field(..., alias="TechSupport", description="Yes, No, or No internet service")
    streaming_tv: str = Field(..., alias="StreamingTV", description="Yes, No, or No internet service")
    streaming_movies: str = Field(..., alias="StreamingMovies", description="Yes, No, or No internet service")
    contract: str = Field(..., alias="Contract", description="Contract type")
    paperless_billing: str = Field(..., alias="PaperlessBilling", description="Yes or No")
    payment_method: str = Field(..., alias="PaymentMethod", description="Payment method")

    # ── Validators ────────────────────────────────────────────
    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        allowed = {"Month-to-month", "One year", "Two year"}
        if v not in allowed:
            raise ValueError(f"contract must be one of {allowed}, got '{v}'")
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        allowed = {"DSL", "Fiber optic", "No"}
        if v not in allowed:
            raise ValueError(f"internet_service must be one of {allowed}, got '{v}'")
        return v


class PredictionResult(BaseModel):
    """Output schema — documents and validates what your pipeline returns."""

    churn_prediction: bool
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    customer_id: Optional[str] = None
    model_version: Optional[str] = None