"""Application configuration using Pydantic v2 Settings.

This module provides type-safe, validated configuration management with
environment variable support and clear separation of concerns.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    """Project paths configuration.

    All paths are resolved relative to the project root, which is automatically
    determined as the parent directory of the src/ folder.
    """

    @computed_field  # type: ignore[prop-decorator]
    @property
    def project_root(self) -> Path:
        """Project root directory (parent of src/).

        :return: Absolute path to project root
        """
        return Path(__file__).resolve().parent.parent

    @computed_field  # type: ignore[prop-decorator]
    @property
    def src_dir(self) -> Path:
        """Source code directory.

        :return: Absolute path to src directory
        """
        return self.project_root / "src"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def datasets_dir(self) -> Path:
        """Datasets directory.

        :return: Absolute path to datasets directory
        """
        return self.project_root / "datasets"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def samples_dir(self) -> Path:
        """Model weights and samples directory.

        :return: Absolute path to samples directory
        """
        return self.project_root / "samples"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def docs_dir(self) -> Path:
        """Documentation directory.

        :return: Absolute path to docs directory
        """
        return self.project_root / "docs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def images_dir(self) -> Path:
        """Raw images directory.

        :return: Absolute path to images directory
        """
        return self.datasets_dir / "images"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def labels_dir(self) -> Path:
        """Label files directory.

        :return: Absolute path to labels directory
        """
        return self.datasets_dir / "labels"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ultralytics_dir(self) -> Path:
        """YOLO-formatted dataset directory.

        :return: Absolute path to ultralytics dataset directory
        """
        return self.datasets_dir / "ultralytics"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_config(self) -> Path:
        """Path to data.yaml configuration file.

        :return: Absolute path to data.yaml
        """
        return self.project_root / "data.yaml"


class ModelWeightsConfig(BaseModel):
    """Model weights paths configuration.

    Paths can be absolute or relative to the project root.
    """

    yolo_weights: Path = Field(
        default=Path("samples/best.pt"),
        description="Path to YOLOv8 model weights",
    )
    faster_rcnn_weights: Path = Field(
        default=Path("samples/fasterrcnn_best.pt"),
        description="Path to Faster R-CNN model weights",
    )
    detr_weights: Path = Field(
        default=Path("samples/detr_best.pt"),
        description="Path to DETR model weights",
    )
    detr_model_name: str = Field(
        default="facebook/detr-resnet-50",
        description="Hugging Face DETR model identifier",
    )

    @model_validator(mode="after")
    def resolve_paths(self) -> "ModelWeightsConfig":
        """Resolve weight paths to absolute paths.

        :return: Self with resolved paths
        """
        project_root = Path(__file__).resolve().parent.parent

        if not self.yolo_weights.is_absolute():
            self.yolo_weights = project_root / self.yolo_weights
        if not self.faster_rcnn_weights.is_absolute():
            self.faster_rcnn_weights = project_root / self.faster_rcnn_weights
        if not self.detr_weights.is_absolute():
            self.detr_weights = project_root / self.detr_weights

        return self


class TrainingConfig(BaseModel):
    """Model training hyperparameters configuration."""

    train_ratio: float = Field(
        default=0.7,
        ge=0.1,
        le=0.9,
        description="Training set ratio (0.1-0.9)",
    )
    val_ratio: float = Field(
        default=0.3,
        ge=0.1,
        le=0.9,
        description="Validation set ratio (0.1-0.9)",
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Training batch size",
    )
    epochs: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Number of training epochs",
    )
    learning_rate: float = Field(
        default=0.005,
        gt=0,
        le=1,
        description="Initial learning rate",
    )
    image_size: int = Field(
        default=640,
        ge=320,
        le=1280,
        description="Input image size (pixels)",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def test_ratio(self) -> float:
        """Calculate test set ratio from train and validation ratios.

        :return: Test ratio (1.0 - train_ratio - val_ratio)
        """
        return 1.0 - self.train_ratio - self.val_ratio

    def model_post_init(self, __context: object) -> None:
        """Validate that train and val ratios sum to <= 1.0.

        :raises ValueError: If ratios sum exceeds 1.0
        """
        total = self.train_ratio + self.val_ratio
        if total > 1.0:
            msg = (
                f"train_ratio ({self.train_ratio}) + "
                f"val_ratio ({self.val_ratio}) = {total} > 1.0"
            )
            raise ValueError(msg)


class ScraperConfig(BaseModel):
    """Web scraper configuration for camera image collection."""

    csv_file: Path = Field(
        default=Path("datasets/cctv-aware-jyvaskyla.csv"),
        description="CSV file with camera locations",
    )
    output_dir: Path = Field(
        default=Path("datasets/screenshots"),
        description="Output directory for screenshots",
    )
    browser_timeout_ms: int = Field(
        default=20000,
        ge=1000,
        le=60000,
        description="Browser page load timeout in milliseconds",
    )
    cookie_dialog_timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Cookie dialog timeout in milliseconds",
    )
    page_settle_timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Page settle timeout in milliseconds",
    )
    headless: bool = Field(
        default=True,
        description="Run browser in headless mode",
    )
    reject_all_text: str = Field(
        default="Reject all",
        description="English cookie rejection button text",
    )
    reject_all_text_gr: str = Field(
        default="Απόρριψη όλων",
        description="Greek cookie rejection button text",
    )

    @model_validator(mode="after")
    def resolve_paths(self) -> "ScraperConfig":
        """Resolve scraper paths to absolute paths.

        :return: Self with resolved paths
        """
        project_root = Path(__file__).resolve().parent.parent

        if not self.csv_file.is_absolute():
            self.csv_file = project_root / self.csv_file
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir

        return self


class Settings(BaseSettings):
    """Main application settings.

    Configuration can be overridden via environment variables with CCTV_ prefix.
    Nested settings use double underscore delimiter.

    Examples:
        CCTV_LOG_LEVEL=DEBUG
        CCTV_TRAINING__BATCH_SIZE=8
        CCTV_TRAINING__LEARNING_RATE=0.001
        CCTV_MODELS__YOLO_WEIGHTS=/custom/path/model.pt
    """

    model_config = SettingsConfigDict(
        env_prefix="CCTV_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelWeightsConfig = Field(default_factory=ModelWeightsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def uppercase_log_level(cls, v: str) -> str:
        """Convert log level to uppercase to support case-insensitive input.

        :param v: Log level string
        :return: Uppercase log level
        """
        if isinstance(v, str):
            return v.upper()
        return v


# Global settings instance - singleton pattern
settings = Settings()
