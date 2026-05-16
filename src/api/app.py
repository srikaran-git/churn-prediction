# src/api/app.py

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.models.predictor import ChurnPredictor
from src.models.schemas import CustomerFeatures, PredictionResult
from src.utils.config_loader import load_config
from src.utils.exceptions import PredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

predictor = ChurnPredictor(
    model_path=config["paths"]["model_output"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up — loading pipeline...")
    predictor.load()
    logger.info("Pipeline loaded. API ready.")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn from Telco data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    logger.info("Root endpoint called")
    return {"message": "Churn Prediction API is running"}


@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(features: CustomerFeatures) -> PredictionResult:
    logger.info("POST /predict called")
    try:
        return predictor.predict(features)
    except PredictionError as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))