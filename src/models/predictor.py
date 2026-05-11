# src/models/predictor.py
"""
Prediction module — loads a saved model and predicts on new data.

Responsibility: ONLY making predictions on new input data.
"""

# ── Standard library ──────────────────────────────────────────
from typing import Dict

# ── Third-party ───────────────────────────────────────────────
import pandas as pd

# ── Internal ──────────────────────────────────────────────────
from src.models.trainer import load_model
from src.utils.exceptions import PredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_single(
    model_path: str,
    input_data: Dict,
) -> Dict:
    """
    Make a prediction for a single customer.

    Args:
        model_path : Path to the saved model .pkl file.
        input_data : Dictionary of feature name → value.

    Returns:
        Dictionary with keys:
            - 'prediction' : 0 or 1
            - 'probability': float between 0 and 1
            - 'label'      : 'Will Churn' or 'Will Not Churn'

    Raises:
        PredictionError: If prediction or probability inference fails.
    """
    model = load_model(model_path)
    df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(df)[0]
    except Exception as e:
        logger.error("Error during single prediction: %s", e)
        raise PredictionError("Failed to make prediction.") from e

    try:
        probability = model.predict_proba(df)[0][1]
    except Exception as e:
        logger.error("Error during probability prediction: %s", e)
        raise PredictionError("Failed to make probability prediction.") from e

    label = "Will Churn" if prediction == 1 else "Will Not Churn"

    logger.info(
        "Single prediction complete | label=%s | probability=%.4f",
        label,
        probability,
    )

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "label": label,
    }


def predict_batch(
    model_path: str,
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Make predictions for a batch of customers.

    Args:
        model_path : Path to saved model .pkl file.
        input_df   : DataFrame with multiple customer rows.

    Returns:
        Original DataFrame with two new columns added:
            - 'prediction'  : 0 or 1
            - 'probability' : churn probability score

    Raises:
        PredictionError: If prediction or probability inference fails.
    """
    model = load_model(model_path)

    # Work on a copy — never mutate the caller's DataFrame
    result_df = input_df.copy()

    try:
        predictions = model.predict(input_df)
    except Exception as e:
        logger.error("Error during batch prediction: %s", e)
        raise PredictionError("Failed to make batch predictions.") from e

    try:
        probabilities = model.predict_proba(input_df)[:, 1]
    except Exception as e:
        logger.error("Error during batch probability prediction: %s", e)
        raise PredictionError("Failed to make batch probability predictions.") from e

    result_df["prediction"] = predictions
    result_df["probability"] = probabilities.round(4)

    logger.info(
        "Batch prediction complete | rows=%d | churn_rate=%.2f%%",
        len(result_df),
        result_df["prediction"].mean() * 100,
    )

    return result_df
