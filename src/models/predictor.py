# src/models/predictor.py
"""
Prediction module — loads a saved model and predicts on new data.

Responsibility: ONLY making predictions on new input data.
"""

from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.trainer import load_model
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
            - 'prediction': 0 or 1
            - 'probability': float between 0 and 1
            - 'label': 'Will Churn' or 'Will Not Churn'

    YOUR TASK:
        1. Load model using load_model(model_path)
        2. Convert input_data dict → single-row DataFrame
           Hint: pd.DataFrame([input_data])
        3. Get prediction: model.predict(df)[0]
        4. Get probability: model.predict_proba(df)[0][1]
        5. Return dict with prediction, probability, label
           label = 'Will Churn' if prediction == 1
                   else 'Will Not Churn'
    """
    model = load_model(model_path)
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    label = "Will Churn" if prediction == 1 else "Will Not Churn"

    return {"prediction": prediction, "probability": probability, "label": label}


def predict_batch(
    model_path: str,
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Make predictions for a batch of customers.

    Args:
        model_path : Path to saved model.
        input_df   : DataFrame with multiple customer rows.

    Returns:
        Original DataFrame with two new columns added:
            - 'prediction'  : 0 or 1
            - 'probability' : churn probability score

    YOUR TASK:
        1. Load model
        2. Get predictions for all rows
        3. Get probabilities for all rows
        4. Add both as new columns to input_df copy
        5. Return the new DataFrame
    """
    model = load_model(model_path)
    df = input_df.copy()
    df["prediction"] = model.predict(df)
    df["probability"] = model.predict_proba(df)[:, 1]
    return df
