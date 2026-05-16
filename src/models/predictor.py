# src/models/predictor.py

import pandas as pd

from src.models.model_loader import load_pipeline
from src.models.schemas import CustomerFeatures, PredictionResult
from src.utils.exceptions import PredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChurnPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None

    def load(self) -> None:
        """Load the pipeline from disk. Call once at startup."""
        self.pipeline = load_pipeline(self.model_path)
        logger.info(f"Pipeline loaded from {self.model_path}")

    def predict(self, features: CustomerFeatures) -> PredictionResult:
        """Run prediction on a single customer."""
        if self.pipeline is None:
            raise PredictionError("Model not loaded. Call load() first.")

        input_df = pd.DataFrame([features.model_dump(by_alias=True)])
        logger.info(f"Running prediction for input: {input_df.to_dict(orient='records')}")

        prediction = self.pipeline.predict(input_df)[0]
        probability = self.pipeline.predict_proba(input_df)[0][1]

        result = PredictionResult(
            churn_prediction=bool(prediction),
            churn_probability=round(float(probability), 4),
        )

        logger.info(f"Prediction result: {result}")
        return result