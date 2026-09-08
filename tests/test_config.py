"""Tests for Pydantic v2 configuration settings.

Field bounds are pydantic's job; they are checked once each in a single
parametrized test. The hand-written validators and the env-var plumbing get
individual tests.
"""

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
        """project_root resolves to the absolute repository directory."""
        paths = PathsConfig()
        assert paths.project_root.name == "cctv_detection"
        assert paths.project_root.is_absolute()

    def test_derived_paths_hang_off_project_root(self) -> None:
        """Every derived path is anchored under project_root."""
        paths = PathsConfig()

        assert paths.src_dir == paths.project_root / "src"
        assert paths.datasets_dir == paths.project_root / "datasets"
        assert paths.samples_dir == paths.project_root / "samples"
        assert paths.docs_dir == paths.project_root / "docs"
        assert paths.images_dir == paths.datasets_dir / "images"
        assert paths.labels_dir == paths.datasets_dir / "labels"
        assert paths.ultralytics_dir == paths.datasets_dir / "ultralytics"
        assert paths.data_config == paths.project_root / "data.yaml"


class TestModelWeightsConfig:
    """Tests for ModelWeightsConfig."""

    def test_default_weights_paths(self) -> None:
        """Default weight locations match the documented layout."""
        models = ModelWeightsConfig()

        assert models.yolo_weights.name == "best.pt"
        assert models.faster_rcnn_weights.name == "fasterrcnn_best.pt"
        assert models.detr_weights.name == "final"  # DETR uses directory format

    def test_relative_paths_are_resolved(self) -> None:
        """Relative defaults are resolved to absolute paths by the validator."""
        models = ModelWeightsConfig()

        assert models.yolo_weights.is_absolute()
        assert models.faster_rcnn_weights.is_absolute()
        assert models.detr_weights.is_absolute()

    def test_absolute_paths_are_preserved(self) -> None:
        """An absolute path passes through the validator unchanged."""
        absolute_path = Path("/custom/path/model.pt")
        models = ModelWeightsConfig(yolo_weights=absolute_path)

        assert models.yolo_weights == absolute_path


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self) -> None:
        """Documented default hyperparameters are pinned."""
        training = TrainingConfig()

        assert training.train_ratio == 0.7
        assert training.val_ratio == 0.3
        assert training.batch_size == 4
        assert training.epochs == 20
        assert training.learning_rate == 0.005
        assert training.image_size == 640

    def test_test_ratio_computation(self) -> None:
        """test_ratio is whatever train and val leave over."""
        training = TrainingConfig(train_ratio=0.7, val_ratio=0.2)

        assert training.test_ratio == pytest.approx(0.1)

    def test_invalid_ratios_sum_exceeds_one(self) -> None:
        """The model validator rejects train + val above 1.0."""
        with pytest.raises(ValueError) as exc_info:
            TrainingConfig(train_ratio=0.7, val_ratio=0.5)

        assert "1.2 > 1.0" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"train_ratio": 0.05}, "greater than or equal to 0.1"),
            ({"train_ratio": 0.95}, "less than or equal to 0.9"),
            ({"val_ratio": 0.05}, "greater than or equal to 0.1"),
            ({"val_ratio": 0.95}, "less than or equal to 0.9"),
            ({"batch_size": 0}, "greater than or equal to 1"),
            ({"batch_size": 200}, "less than or equal to 128"),
            ({"epochs": 0}, "greater than or equal to 1"),
            ({"epochs": 1001}, "less than or equal to 1000"),
            ({"learning_rate": 0}, "greater than 0"),
            ({"learning_rate": 1.5}, "less than or equal to 1"),
            ({"image_size": 256}, "greater than or equal to 320"),
            ({"image_size": 1920}, "less than or equal to 1280"),
        ],
    )
    def test_field_bounds_are_enforced(self, overrides: dict, message: str) -> None:
        """Each hyperparameter rejects values outside its declared bounds."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**overrides)

        assert message in str(exc_info.value)


class TestScraperConfig:
    """Tests for ScraperConfig."""

    def test_default_values(self) -> None:
        """Documented scraper defaults are pinned."""
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
        """Relative defaults are resolved to absolute paths by the validator."""
        scraper = ScraperConfig()

        assert scraper.csv_file.is_absolute()
        assert scraper.output_dir.is_absolute()

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"browser_timeout_ms": 500}, "greater than or equal to 1000"),
            ({"browser_timeout_ms": 100000}, "less than or equal to 60000"),
            ({"cookie_dialog_timeout_ms": 500}, "greater than or equal to 1000"),
            ({"cookie_dialog_timeout_ms": 40000}, "less than or equal to 30000"),
            ({"page_settle_timeout_ms": 500}, "greater than or equal to 1000"),
            ({"page_settle_timeout_ms": 40000}, "less than or equal to 30000"),
        ],
    )
    def test_timeout_bounds_are_enforced(self, overrides: dict, message: str) -> None:
        """Each timeout rejects values outside its declared bounds."""
        with pytest.raises(ValidationError) as exc_info:
            ScraperConfig(**overrides)

        assert message in str(exc_info.value)


class TestSettings:
    """Tests for the root Settings class and its env-var plumbing."""

    def test_env_override_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CCTV_ prefix maps env vars onto top-level fields."""
        monkeypatch.setenv("CCTV_LOG_LEVEL", "DEBUG")

        assert Settings().log_level == "DEBUG"

    def test_env_override_nested_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The __ delimiter maps env vars onto nested config fields."""
        monkeypatch.setenv("CCTV_TRAINING__BATCH_SIZE", "16")

        assert Settings().training.batch_size == 16

    def test_invalid_log_level(self) -> None:
        """log_level is restricted to the documented literal values."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(log_level="INVALID")  # type: ignore[arg-type]

        assert "log_level" in str(exc_info.value)

    def test_case_insensitive_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env var names are case-insensitive and log_level is upper-cased."""
        monkeypatch.setenv("cctv_log_level", "warning")

        assert Settings().log_level == "WARNING"
