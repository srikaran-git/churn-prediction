import pytest
from sklearn.pipeline import Pipeline

from src.models.model_loader import get_model_info, load_pipeline, save_pipeline
from src.models.pipeline_builder import build_pipeline
from src.models.trainer import ModelTrainer
from src.utils.exceptions import ModelLoadingError

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def trained_trainer(clean_churn_df):
    """Return a ModelTrainer that has already been trained."""
    trainer = ModelTrainer()
    trainer.train(clean_churn_df)
    return trainer


# ─────────────────────────────────────────────
# ModelTrainer tests
# ─────────────────────────────────────────────


class TestModelTrainer:

    def test_train_returns_self(self, clean_churn_df):
        """train() should return self to allow method chaining."""
        trainer = ModelTrainer()
        result = trainer.train(clean_churn_df)
        assert result is trainer

    def test_get_model_returns_pipeline(self, trained_trainer):
        """get_model() should return a fitted sklearn Pipeline."""
        model = trained_trainer.get_model()
        assert isinstance(model, Pipeline)

    def test_get_model_before_training_raises(self):
        """get_model() before train() should raise an exception."""
        trainer = ModelTrainer()
        with pytest.raises(Exception):
            trainer.get_model()

    def test_metrics_contain_expected_keys(self, trained_trainer):
        """get_metrics() should return accuracy, f1, and roc_auc."""
        metrics = trained_trainer.get_metrics()
        assert "accuracy" in metrics
        assert "f1" in metrics
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
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_preprocessor_step(self):
        """Pipeline must contain a preprocessor step."""
        pipeline = build_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names

    def test_pipeline_has_classifier_step(self):
        """Pipeline must contain a classifier step."""
        pipeline = build_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert "classifier" in step_names

    def test_pipeline_can_fit(self, clean_churn_df):
        """Pipeline should fit on the clean dataframe without errors."""
        from src.data.data_loader import load_data

        pipeline = build_pipeline()
        X = clean_churn_df.drop(columns=["Churn"])
        y = clean_churn_df["Churn"].map({"Yes": 1, "No": 0})
        pipeline.fit(X, y)  # should not raise


# ─────────────────────────────────────────────
# model_loader tests
# ─────────────────────────────────────────────


class TestModelLoader:

    def test_save_creates_file(self, trained_trainer, tmp_path):
        """save_pipeline() should write a file to the given path."""
        output_path = tmp_path / "test_pipeline.pkl"
        save_pipeline(trained_trainer.get_model(), str(output_path))
        assert output_path.exists()

    def test_load_returns_pipeline(self, trained_trainer, tmp_path):
        """load_pipeline() should return a sklearn Pipeline."""
        output_path = tmp_path / "test_pipeline.pkl"
        save_pipeline(trained_trainer.get_model(), str(output_path))

        loaded = load_pipeline(str(output_path))
        assert isinstance(loaded, Pipeline)

    def test_load_missing_file_raises(self, tmp_path):
        """load_pipeline() should raise ModelLoadingError for missing files."""
        bad_path = str(tmp_path / "does_not_exist.pkl")
        with pytest.raises(ModelLoadingError):
            load_pipeline(bad_path)

    def test_get_model_info_returns_expected_keys(self, trained_trainer, tmp_path):
        """get_model_info() should return a dict with required metadata keys."""
        output_path = tmp_path / "test_pipeline.pkl"
        save_pipeline(trained_trainer.get_model(), str(output_path))

        info = get_model_info(str(output_path))
        assert "model_type" in info
        assert "file_size_kb" in info
        assert "created_at" in info
