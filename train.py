# train.py
"""
Main training script for the Churn Prediction model.

This script orchestrates the full training pipeline:
    1. Load raw data
    2. Preprocess and split
    3. Train model
    4. Evaluate performance
    5. Save model artifact

Run with:
    python train.py
"""

# ── Standard library ──────────────────────────────────────────
import sys

# ── Internal ──────────────────────────────────────────────────
from src.data.data_loader import get_data_summary, load_csv
from src.data.preprocessor import run_preprocessing_pipeline
from src.models.evaluater import (
    evaluate_model,
    log_classification_report,
    log_confusion_matrix,
    log_metrics,
    log_top_features,
)
from src.models.trainer import save_model, train_random_forest
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline() -> None:
    """
    Execute the complete model training pipeline.

    Steps:
        1. Load config
        2. Load raw data
        3. Validate + preprocess data
        4. Train model
        5. Evaluate model
        6. Save model
    """
    logger.info("=" * 50)
    logger.info("CHURN PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 50)

    # ── Step 1: Load config ────────────────────────────────────
    logger.info("[1/6] Loading configuration...")
    config = load_config()

    raw_data_path = config["data"]["raw_path"]
    model_save_path = config["model"]["save_path"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    model_params = {
        "n_estimators": config["model"]["parameters"]["n_estimators"],
        "max_depth": config["model"]["parameters"]["max_depth"],
        "random_state": random_state,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # ── Step 2: Load data ──────────────────────────────────────
    logger.info("[2/6] Loading raw data...")
    df = load_csv(raw_data_path)
    get_data_summary(df)

    # ── Step 3: Preprocess ─────────────────────────────────────
    logger.info("[3/6] Preprocessing data...")
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    # ── Step 4: Train ──────────────────────────────────────────
    logger.info("[4/6] Training model...")
    model = train_random_forest(X_train, y_train, model_params)

    # ── Step 5: Evaluate ───────────────────────────────────────
    logger.info("[5/6] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    log_metrics(metrics)
    log_classification_report(model, X_test, y_test)
    log_confusion_matrix(model, X_test, y_test)
    log_top_features(model, X_train.columns.tolist())

    # ── Step 6: Save ───────────────────────────────────────────
    logger.info("[6/6] Saving model...")
    save_model(model, model_save_path)

    logger.info("=" * 50)
    logger.info("TRAINING PIPELINE COMPLETE ✓")
    logger.info(f"Model saved to : {model_save_path}")
    logger.info(f"F1 Score       : {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC        : {metrics['roc_auc']:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    try:
        run_training_pipeline()
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
