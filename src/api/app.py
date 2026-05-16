from fastapi import FastAPI

from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn from Telco data.",
    version="1.0.0",
)


@app.get("/")
def root():
    logger.info("Root endpoint called")
    return {"message": "Churn Prediction API is running"}


@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "ok"}