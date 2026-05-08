# train.py  (relevant section — replace your existing model training block)
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.model_loader import get_model_info, save_pipeline
from src.models.pipeline_builder import build_pipeline
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training():
    config = load_config()

    # 1. Load raw data (no manual preprocessing needed anymore)
    from src.data.data_loader import load_data

    X_train, X_test, y_train, y_test = load_data(config)

    # 2. Build pipeline
    model = RandomForestClassifier(
        n_estimators=config["model"]["parameters"]["n_estimators"],
        max_depth=config["model"]["parameters"]["max_depth"],
        random_state=config["data"]["random_state"],
    )
    pipeline = build_pipeline(model)

    # 3. Fit — preprocessing + model training in one call
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline training complete")

    # 4. Evaluate
    from src.models.evaluator import evaluate_model

    metrics = evaluate_model(pipeline, X_test, y_test)
    logger.info("Metrics: %s", metrics)

    # 5. Save
    output_path = config["paths"]["model_output"]
    save_pipeline(pipeline, output_path)

    info = get_model_info(pipeline)
    logger.info("Model info: %s", info)


if __name__ == "__main__":
    run_training()
