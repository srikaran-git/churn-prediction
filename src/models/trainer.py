from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger


class ModelTrainer:
    """Encapsulates model training and evaluation for churn prediction."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.metrics = {}
        self.logger = get_logger(__name__)

    def train(self, X_train, y_train):
        """Train the model and store it on self.model."""
        try:
            self.logger.info("Initializing model...")
            params = self.config.get("model", {}).get("parameters", {})
            self.model = RandomForestClassifier(**params)

            self.logger.info("Training started...")
            self.model.fit(X_train, y_train)
            self.logger.info("Training complete.")

        except Exception as e:
            self.logger.error("Training failed.", exc_info=True)
            raise ModelTrainingError("Model training failed.") from e

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate the trained model and return metrics."""
        if self.model is None:
            raise ModelTrainingError("Model not trained. Call train() first.")

        try:
            preds = self.model.predict(X_test)
            proba = self.model.predict_proba(X_test)[:, 1]

            self.metrics = {
                "accuracy": round(accuracy_score(y_test, preds), 4),
                # "f1_score": round(f1_score(y_test, preds), 4),
                "roc_auc": round(roc_auc_score(y_test, proba), 4),
            }

            for name, value in self.metrics.items():
                self.logger.info("  %s: %s", name, value)

            return self.metrics

        except Exception as e:
            self.logger.error("Evaluation failed.", exc_info=True)
            raise ModelTrainingError("Model evaluation failed.") from e

    def get_model(self):
        """Return the trained model object."""
        if self.model is None:
            raise ModelTrainingError("No trained model available.")
        return self.model

    def get_metrics(self) -> dict:
        """Return the last computed metrics."""
        return self.metrics
