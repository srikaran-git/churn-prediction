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

    Example:
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> print(metrics["f1_score"])
    """
    logger.info("Evaluating model on test set...")

    # Get predictions (class labels)
    y_pred = model.predict(X_test)

    # Get probabilities (for ROC-AUC)
    y_proba = model.predict_proba(X_test)[:, 1]

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
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1 Score  : {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    logger.info("─" * 40)


def log_classification_report(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Log the full sklearn classification report.

    Args:
        model  : Trained classifier.
        X_test : Test features.
        y_test : True labels.
    """
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not Churned", "Churned"],
    )
    logger.info(f"Classification Report:\n{report}")


def log_confusion_matrix(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Log the confusion matrix in a readable format.

    Args:
        model  : Trained classifier.
        X_test : Test features.
        y_test : True labels.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Confusion Matrix:")
    logger.info(f"                 Predicted No  Predicted Yes")
    logger.info(f"  Actual No    :     {cm[0][0]:<10}  {cm[0][1]}")
    logger.info(f"  Actual Yes   :     {cm[1][0]:<10}  {cm[1][1]}")


def log_top_features(
    model: RandomForestClassifier,
    feature_names: list,
    top_n: int = 10,
) -> None:
    """
    Log the top N most important features used by the model.

    Args:
        model        : Trained RandomForestClassifier.
        feature_names: List of feature column names.
        top_n        : Number of top features to show.

    HINT:
        - model.feature_importances_ gives importance scores
        - zip() them with feature_names
        - Sort by importance descending
        - Log top_n results
    """
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    logger.info("Top %d Most Important Features:", top_n)
    for feature, importance in feature_importance[:top_n]:
        logger.info("  %s: %.4f", feature, importance)
