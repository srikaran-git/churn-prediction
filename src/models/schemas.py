# src/models/schemas.py

from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """Input schema — validates one customer record before it reaches the model."""

    tenure: int = Field(..., ge=0, description="Months as a customer")
    monthly_charges: float = Field(..., gt=0, description="Monthly bill")
    total_charges: float = Field(..., ge=0, description="Total billed to date")
    contract: str = Field(..., description="Contract type")
    payment_method: str = Field(..., description="Payment method")
    internet_service: str = Field(..., description="Internet service type")

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

    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: bool
    model_version: str
