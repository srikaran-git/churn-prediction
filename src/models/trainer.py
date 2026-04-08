# src/models/trainer.py
"""
Model training module for the churn prediction pipeline.

Responsibility: ONLY training the model and saving it.
No data loading, no evaluation — just training.
"""

# ── Standard library ──────────────────────────────────────────
from pathlib import Path
from typing import Any, Dict

# ── Third-party ───────────────────────────────────────────────
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ── Internal ──────────────────────────────────────────────────
from src.utils.logger import get_logger

logger = get_logger(__name__)
# ── Constants ─────────────────────────────────────────────────
DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,  # use all CPU cores
    "class_weight": "balanced",  # handles class imbalance
}


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict[str, Any] = DEFAULT_MODEL_PARAMS,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the provided training data.

    Args:
        X_train: Training features.
        y_train: Training target variable.
        model_params: Hyperparameters for the Random Forest model.

    Returns:
        Trained Random Forest model.
    """
    logger.info(f"Training Random Forest with parameters: {model_params}")
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model


def save_model(model: RandomForestClassifier, save_path: str) -> None:
    """
    Save the trained model to disk using joblib.

    Args:
        model: Trained Random Forest model to save.
        save_path: File path where the model should be saved.

    Raises:
        IOError: If there is an issue saving the model to disk.
    Example:
        >>> save_model(model, "models/model_v1.pkl")
    """
    path = Path(save_path)
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved successfully at {save_path}")
    except Exception as e:
        logger.error(f"Failed to save model at {save_path}: {e}")
        raise IOError(f"Failed to save model at {save_path}: {e}")


def load_model(model_path: str) -> RandomForestClassifier:
    """
    Load a trained model from disk.

    Args:
        model_path: File path to the saved model.

    Returns:
        Loaded Random Forest model.

    Raises:
        IOError: If there is an issue loading the model from disk.

    Example:
        >>> model = load_model("models/model_v1.pkl")
        >>> predictions = model.predict(X_test)
    """
    path = Path(model_path)
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise IOError(f"Failed to load model from {model_path}: {e}")
