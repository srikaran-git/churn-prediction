import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.model_loader import get_model_info, load_pipeline, save_pipeline
from src.models.pipeline_builder import build_pipeline
from src.models.trainer import ModelTrainer
from src.utils.config_loader import load_config
from src.utils.exceptions import ModelLoadError

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def config():
    """Load the project config."""
    return load_config()


@pytest.fixture
def trained_trainer(config, clean_churn_df):
    """Return a ModelTrainer that has been trained and evaluated."""
    trainer = ModelTrainer(config)
    numeric_features = config["features"]["numeric"]
    X = clean_churn_df[numeric_features].fillna(0)
    y = clean_churn_df["Churn"].map({"Yes": 1, "No": 0})
    trainer.train(X, y)
    trainer.evaluate(X, y)
    return trainer


# ─────────────────────────────────────────────
# ModelTrainer tests
# ─────────────────────────────────────────────


class TestModelTrainer:

    def test_train_sets_model(self, config, clean_churn_df):
        """After train(), get_model() should not be None."""
        trainer = ModelTrainer(config)
        numeric_features = config["features"]["numeric"]
        X = clean_churn_df[numeric_features].fillna(0)
        y = clean_churn_df["Churn"].map({"Yes": 1, "No": 0})
        trainer.train(X, y)
        assert trainer.get_model() is not None

    def test_get_model_returns_random_forest(self, trained_trainer):
        """get_model() should return a RandomForestClassifier."""
        model = trained_trainer.get_model()
        assert isinstance(model, RandomForestClassifier)

    def test_get_model_before_training_raises(self, config):
        """get_model() before train() should raise an exception."""
        trainer = ModelTrainer(config)
        with pytest.raises(Exception):
            trainer.get_model()

    def test_evaluate_before_training_raises(self, config):
        """evaluate() before train() should raise ModelTrainingError."""
        from src.utils.exceptions import ModelTrainingError

        trainer = ModelTrainer(config)
        with pytest.raises(ModelTrainingError):
            trainer.evaluate([[1, 2, 3]], [0])

    def test_metrics_contain_expected_keys(self, trained_trainer):
        """get_metrics() should return accuracy, f1, and roc_auc."""
        metrics = trained_trainer.get_metrics()
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

    def test_metrics_are_valid_scores(self, trained_trainer):
        """All metric values should be floats between 0 and 1."""
        metrics = trained_trainer.get_metrics()
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"


# ─────────────────────────────────────────────
# pipeline_builder tests
# ─────────────────────────────────────────────


class TestBuildPipeline:

    def test_returns_sklearn_pipeline(self):
        """build_pipeline() should return a sklearn Pipeline instance."""
        pipeline = build_pipeline(RandomForestClassifier())
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_preprocessor_step(self):
        """Pipeline must contain a preprocessor step."""
        pipeline = build_pipeline(RandomForestClassifier())
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names

    def test_pipeline_has_model_step(self):
        """Pipeline must contain a model step."""
        pipeline = build_pipeline(RandomForestClassifier())
        step_names = [name for name, _ in pipeline.steps]
        assert "model" in step_names

    def test_pipeline_can_fit(self, clean_churn_df):
        """Pipeline should fit on the clean dataframe without errors."""
        pipeline = build_pipeline(RandomForestClassifier())
        X = clean_churn_df.drop(columns=["Churn"])
        y = clean_churn_df["Churn"].map({"Yes": 1, "No": 0})
        pipeline.fit(X, y)


# ─────────────────────────────────────────────
# model_loader tests
# ─────────────────────────────────────────────


class TestModelLoader:

    def test_save_creates_file(self, tmp_path):
        """save_pipeline() should write a file to the given path."""
        pipeline = build_pipeline(RandomForestClassifier())
        output_path = tmp_path / "test_model.pkl"
        save_pipeline(pipeline, str(output_path))
        assert output_path.exists()

    def test_load_returns_correct_type(self, tmp_path):
        """load_pipeline() should return a sklearn Pipeline."""
        pipeline = build_pipeline(RandomForestClassifier())
        output_path = tmp_path / "test_model.pkl"
        save_pipeline(pipeline, str(output_path))
        loaded = load_pipeline(str(output_path))
        assert isinstance(loaded, Pipeline)

    def test_load_missing_file_raises(self, tmp_path):
        """load_pipeline() should raise ModelLoadError for missing files."""
        bad_path = str(tmp_path / "does_not_exist.pkl")
        with pytest.raises(ModelLoadError):
            load_pipeline(bad_path)

    def test_get_model_info_returns_expected_keys(self, tmp_path):
        """get_model_info() should return a dict with required metadata keys."""
        pipeline = build_pipeline(RandomForestClassifier())
        output_path = tmp_path / "test_model.pkl"
        save_pipeline(pipeline, str(output_path))
        loaded = load_pipeline(str(output_path))
        info = get_model_info(loaded)
        assert "model_type" in info
        assert "steps" in info
        assert "transformers" in info
