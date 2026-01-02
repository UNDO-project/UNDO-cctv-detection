"""Tests for Pydantic v2 configuration settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import (
    ModelWeightsConfig,
    PathsConfig,
    ScraperConfig,
    Settings,
    TrainingConfig,
)


class TestPathsConfig:
    """Tests for PathsConfig."""

    def test_project_root_is_computed(self) -> None:
        """Test that project_root is correctly computed."""
        paths = PathsConfig()
        assert isinstance(paths.project_root, Path)
        assert paths.project_root.name == "cctv_detection"
        assert paths.project_root.is_absolute()

    def test_derived_paths_are_correct(self) -> None:
        """Test that derived paths are computed correctly."""
        paths = PathsConfig()

        assert paths.src_dir == paths.project_root / "src"
        assert paths.datasets_dir == paths.project_root / "datasets"
        assert paths.samples_dir == paths.project_root / "samples"
        assert paths.docs_dir == paths.project_root / "docs"

    def test_dataset_subdirectories(self) -> None:
        """Test that dataset subdirectories are computed correctly."""
        paths = PathsConfig()

        assert paths.images_dir == paths.datasets_dir / "images"
        assert paths.labels_dir == paths.datasets_dir / "labels"
        assert paths.ultralytics_dir == paths.datasets_dir / "ultralytics"

    def test_data_config_path(self) -> None:
        """Test that data_config path is correct."""
        paths = PathsConfig()

        assert paths.data_config == paths.project_root / "data.yaml"


class TestModelWeightsConfig:
    """Tests for ModelWeightsConfig."""

    def test_default_weights_paths(self) -> None:
        """Test that default weight paths are set correctly."""
        models = ModelWeightsConfig()

        assert models.yolo_weights.name == "best.pt"
        assert models.faster_rcnn_weights.name == "fasterrcnn_best.pt"
        assert models.detr_weights.name == "final"  # DETR uses directory format

    def test_relative_paths_are_resolved(self) -> None:
        """Test that relative paths are resolved to absolute paths."""
        models = ModelWeightsConfig()

        # All paths should be absolute
        assert models.yolo_weights.is_absolute()
        assert models.faster_rcnn_weights.is_absolute()
        assert models.detr_weights.is_absolute()

    def test_absolute_paths_are_preserved(self) -> None:
        """Test that absolute paths are not modified."""
        absolute_path = Path("/custom/path/model.pt")
        models = ModelWeightsConfig(yolo_weights=absolute_path)

        assert models.yolo_weights == absolute_path


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self) -> None:
        """Test that default training config values are correct."""
        training = TrainingConfig()

        assert training.train_ratio == 0.7
        assert training.val_ratio == 0.3
        assert training.batch_size == 4
        assert training.epochs == 20
        assert training.learning_rate == 0.005
        assert training.image_size == 640

    def test_test_ratio_computation(self) -> None:
        """Test that test_ratio is computed correctly."""
        training = TrainingConfig(train_ratio=0.7, val_ratio=0.2)

        assert training.test_ratio == pytest.approx(0.1)

    def test_valid_ratios(self) -> None:
        """Test that valid train/val ratios are accepted."""
        # Should not raise
        TrainingConfig(train_ratio=0.6, val_ratio=0.3)
        TrainingConfig(train_ratio=0.8, val_ratio=0.1)

    def test_invalid_train_ratio_too_low(self) -> None:
        """Test that train_ratio < 0.1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(train_ratio=0.05)

        assert "greater than or equal to 0.1" in str(exc_info.value)

    def test_invalid_train_ratio_too_high(self) -> None:
        """Test that train_ratio > 0.9 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(train_ratio=0.95)

        assert "less than or equal to 0.9" in str(exc_info.value)

    def test_invalid_ratios_sum_exceeds_one(self) -> None:
        """Test that train_ratio + val_ratio > 1.0 is rejected."""
        with pytest.raises(ValueError) as exc_info:
            TrainingConfig(train_ratio=0.7, val_ratio=0.5)

        assert "1.2 > 1.0" in str(exc_info.value)

    def test_invalid_batch_size_zero(self) -> None:
        """Test that batch_size = 0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(batch_size=0)

        assert "greater than or equal to 1" in str(exc_info.value)

    def test_invalid_batch_size_too_large(self) -> None:
        """Test that batch_size > 128 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(batch_size=200)

        assert "less than or equal to 128" in str(exc_info.value)

    def test_invalid_epochs_zero(self) -> None:
        """Test that epochs = 0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(epochs=0)

        assert "greater than or equal to 1" in str(exc_info.value)

    def test_invalid_learning_rate_zero(self) -> None:
        """Test that learning_rate <= 0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(learning_rate=0)

        assert "greater than 0" in str(exc_info.value)

    def test_invalid_learning_rate_too_high(self) -> None:
        """Test that learning_rate > 1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(learning_rate=1.5)

        assert "less than or equal to 1" in str(exc_info.value)

    def test_invalid_image_size_too_small(self) -> None:
        """Test that image_size < 320 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(image_size=256)

        assert "greater than or equal to 320" in str(exc_info.value)

    def test_invalid_image_size_too_large(self) -> None:
        """Test that image_size > 1280 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(image_size=1920)

        assert "less than or equal to 1280" in str(exc_info.value)


class TestScraperConfig:
    """Tests for ScraperConfig."""

    def test_default_values(self) -> None:
        """Test that default scraper config values are correct."""
        scraper = ScraperConfig()

        assert scraper.csv_file.name == "cctv-aware-jyvaskyla.csv"
        assert scraper.output_dir.name == "screenshots"
        assert scraper.browser_timeout_ms == 20000
        assert scraper.cookie_dialog_timeout_ms == 5000
        assert scraper.page_settle_timeout_ms == 5000
        assert scraper.headless is True
        assert scraper.reject_all_text == "Reject all"
        assert scraper.reject_all_text_gr == "Απόρριψη όλων"

    def test_paths_are_resolved(self) -> None:
        """Test that scraper paths are resolved to absolute paths."""
        scraper = ScraperConfig()

        assert scraper.csv_file.is_absolute()
        assert scraper.output_dir.is_absolute()

    def test_invalid_browser_timeout_too_low(self) -> None:
        """Test that browser_timeout_ms < 1000 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ScraperConfig(browser_timeout_ms=500)

        assert "greater than or equal to 1000" in str(exc_info.value)

    def test_invalid_browser_timeout_too_high(self) -> None:
        """Test that browser_timeout_ms > 60000 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ScraperConfig(browser_timeout_ms=100000)

        assert "less than or equal to 60000" in str(exc_info.value)


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self) -> None:
        """Test that default settings are created correctly."""
        settings = Settings()

        assert settings.log_level == "INFO"
        assert isinstance(settings.paths, PathsConfig)
        assert isinstance(settings.models, ModelWeightsConfig)
        assert isinstance(settings.training, TrainingConfig)
        assert isinstance(settings.scraper, ScraperConfig)

    def test_env_override_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that CCTV_LOG_LEVEL env var overrides default."""
        monkeypatch.setenv("CCTV_LOG_LEVEL", "DEBUG")

        settings = Settings()

        assert settings.log_level == "DEBUG"

    def test_env_override_nested_training_batch_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that CCTV_TRAINING__BATCH_SIZE env var overrides default."""
        monkeypatch.setenv("CCTV_TRAINING__BATCH_SIZE", "16")

        settings = Settings()

        assert settings.training.batch_size == 16

    def test_env_override_nested_training_learning_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that CCTV_TRAINING__LEARNING_RATE env var overrides default."""
        monkeypatch.setenv("CCTV_TRAINING__LEARNING_RATE", "0.001")

        settings = Settings()

        assert settings.training.learning_rate == pytest.approx(0.001)

    def test_env_override_model_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that CCTV_MODELS__YOLO_WEIGHTS env var overrides default."""
        custom_path = "/custom/path/model.pt"
        monkeypatch.setenv("CCTV_MODELS__YOLO_WEIGHTS", custom_path)

        settings = Settings()

        assert str(settings.models.yolo_weights) == custom_path

    def test_env_override_scraper_headless(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that CCTV_SCRAPER__HEADLESS env var overrides default."""
        monkeypatch.setenv("CCTV_SCRAPER__HEADLESS", "false")

        settings = Settings()

        assert settings.scraper.headless is False

    def test_invalid_log_level(self) -> None:
        """Test that invalid log level is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(log_level="INVALID")  # type: ignore[arg-type]

        assert "log_level" in str(exc_info.value)

    def test_case_insensitive_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables are case-insensitive."""
        monkeypatch.setenv("cctv_log_level", "warning")  # lowercase

        settings = Settings()

        assert settings.log_level == "WARNING"
