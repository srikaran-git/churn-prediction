# src/models/model_loader.py
"""
Handles all model persistence: saving, loading, and validation.
Nothing in the codebase calls joblib directly — it all goes through here.
"""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.utils.exceptions import ModelLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """
    Persist a trained pipeline to disk.

    Args:
        pipeline : Fitted sklearn Pipeline to save.
        path     : Destination file path (e.g. 'models/pipeline_v1.pkl').

    Raises:
        TypeError : If the object passed is not an sklearn Pipeline.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(pipeline).__name__}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, output_path)
    size_kb = output_path.stat().st_size / 1024
    logger.info("Pipeline saved | path=%s | size=%.1f KB", output_path, size_kb)


def load_pipeline(path: str) -> Pipeline:
    """
    Load a trained pipeline from disk with validation.

    Args:
        path: Path to the saved .pkl file.

    Returns:
        Loaded sklearn Pipeline.

    Raises:
        ModelLoadError : If file not found or object is not a Pipeline.
    """
    model_path = Path(path)

    if not model_path.is_file():
        logger.error("Model file not found: %s", model_path)
        raise ModelLoadError(f"No model file at: {model_path}")

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        logger.error("Failed to load model from %s", model_path, exc_info=True)
        raise ModelLoadError(f"Could not load model: {e}") from e

    if not isinstance(pipeline, Pipeline):
        raise ModelLoadError(
            f"Loaded object is not a Pipeline — got {type(pipeline).__name__}"
        )

    steps = [name for name, _ in pipeline.steps]
    logger.info("Pipeline loaded | path=%s | steps=%s", model_path, steps)
    return pipeline


def get_model_info(pipeline: Pipeline) -> dict:
    """
    Extract human-readable metadata from a loaded pipeline.
    Useful for logging at API startup and debugging.

    Args:
        pipeline: Fitted sklearn Pipeline.

    Returns:
        Dict with step names, model type, and feature counts.
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")

    info = {
        "steps": [name for name, _ in pipeline.steps],
        "model_type": type(model).__name__ if model is not None else "unknown",
        "n_features": None,
        "transformers": [],
    }

    if preprocessor and hasattr(preprocessor, "transformers"):
        info["transformers"] = [name for name, _, _ in preprocessor.transformers]
        info["n_features"] = sum(len(cols) for _, _, cols in preprocessor.transformers)

    return info
