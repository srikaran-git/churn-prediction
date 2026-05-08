# src/utils/exceptions.py
class ChurnPredictionError(Exception):
    """Base exception for the entire churn prediction project."""

    pass


class DataValidationError(ChurnPredictionError):
    """Raised when input data fails validation checks."""

    pass


class ModelNotFoundError(ChurnPredictionError):
    """Raised when the model file does not exist at the expected path."""

    pass


class ModelLoadError(ChurnPredictionError):
    """Raised when the model file exists but cannot be loaded."""

    pass


class PredictionError(ChurnPredictionError):
    """Raised when the model fails to generate a prediction."""

    pass


class ConfigurationError(ChurnPredictionError):
    """Raised when required config keys are missing or invalid."""

    pass


class ModelTrainingError(ChurnPredictionError):
    """Raised when model training or evaluation fails."""

    pass
