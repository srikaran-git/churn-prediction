# src/models/pipeline_builder.py

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_pipeline(model):
    """
    Wraps a given sklearn model in a full preprocessing + model pipeline.

    Args:
        model: Any sklearn-compatible estimator (LogisticRegression, etc.)

    Returns:
        sklearn.pipeline.Pipeline
    """
    config = load_config()

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    logger.info(
        "Pipeline built with %d numeric and %d categorical features",
        len(numeric_features),
        len(categorical_features),
    )
    return pipeline
