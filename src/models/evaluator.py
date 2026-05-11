# src/models/evaluator.py
"""
Model evaluation module for the churn prediction pipeline.

Responsibility: ONLY measuring and reporting model performance.
No training, no data loading — just evaluation.
"""

# ── Standard library ──────────────────────────────────────────
from typing import Dict

# ── Third-party ───────────────────────────────────────────────
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Internal ──────────────────────────────────────────────────
from src.utils.exceptions import PredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a trained model and return performance metrics.

    Args:
        model  : Trained sklearn classifier.
        X_test : Test feature matrix.
        y_test : True test labels.

    Returns:
        Dictionary of metric names to scores, e.g.:
        {
            "accuracy"  : 0.81,
            "precision" : 0.67,
            "recall"    : 0.52,
            "f1_score"  : 0.58,
            "roc_auc"   : 0.85,
        }

    Raises:
        PredictionError: If prediction or probability inference fails.

    Example:
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> print(metrics["f1_score"])
    """
    logger.info("Evaluating model on test set...")

    # Get predictions (class labels)
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        logger.error("Error occurred while making predictions: %s", e)
        raise PredictionError("Failed to make predictions.") from e

    # Get probabilities (for ROC-AUC)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        logger.error("Error occurred while making probability predictions: %s", e)
        raise PredictionError("Failed to make probability predictions.") from e

    # Calculate all metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    return metrics


def log_metrics(metrics: Dict[str, float]) -> None:
    """
    Log all evaluation metrics in a readable format.

    Args:
        metrics: Dictionary from evaluate_model().
    """
    logger.info("─" * 40)
    logger.info("MODEL PERFORMANCE")
    logger.info("─" * 40)
    logger.info("  Accuracy  : %.4f", metrics["accuracy"])
    logger.info("  Precision : %.4f", metrics["precision"])
    logger.info("  Recall    : %.4f", metrics["recall"])
    logger.info("  F1 Score  : %.4f", metrics["f1_score"])
    logger.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
    logger.info("─" * 40)


def log_classification_report(
    y_test: pd.Series,
    y_pred: pd.Series,
) -> None:
    """
    Log the full sklearn classification report.

    Args:
        y_test : True labels.
        y_pred : Predicted labels from model.predict().
    """
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not Churned", "Churned"],
    )
    logger.info("Classification Report:\n%s", report)


def log_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series,
) -> None:
    """
    Log the confusion matrix in a readable format.

    Args:
        y_test : True labels.
        y_pred : Predicted labels from model.predict().
    """
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Confusion Matrix:")
    logger.info("                 Predicted No  Predicted Yes")
    logger.info("  Actual No    :     %-10d  %d", cm[0][0], cm[0][1])
    logger.info("  Actual Yes   :     %-10d  %d", cm[1][0], cm[1][1])


def log_top_features(
    model: RandomForestClassifier,
    feature_names: list,
    top_n: int = 10,
) -> None:
    """
    Log the top N most important features used by the model.

    Args:
        model         : Trained RandomForestClassifier.
        feature_names : List of feature column names.
        top_n         : Number of top features to show.
    """
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.info("Top %d Most Important Features:", top_n)
    for feature, importance in feature_importance[:top_n]:
        logger.info("  %s: %.4f", feature, importance)
