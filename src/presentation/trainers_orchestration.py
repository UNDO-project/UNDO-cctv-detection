from loguru import logger

from src.config import settings
from src.infrastructure.dataset_preparer_impl import SklearnDatasetPreparer
from src.infrastructure.trainers import YoloUltralyticsTrainer
from src.infrastructure.device_selector import DeviceSelector


def main() -> None:
    """Train YOLOv8 model on CCTV dataset."""
    # Prepare dataset using paths from settings
    dataset_preparer = SklearnDatasetPreparer()
    dataset_preparer.prepare_ultralytics_dataset(
        source_images=settings.paths.images_dir,
        source_labels=settings.paths.labels_dir,
        output_images=settings.paths.ultralytics_dir / "images",
        output_labels=settings.paths.ultralytics_dir / "labels",
        train_ratio=settings.training.train_ratio,
        val_ratio=settings.training.val_ratio,
        move_files=True,
    )
    logger.info("Dataset is prepared and ready for training.")

    # Initialize trainer with settings
    model_trainer = YoloUltralyticsTrainer(
        model_weights="yolov8n.pt",
        data_config=settings.paths.data_config,
        epochs=settings.training.epochs,
        img_size=settings.training.image_size,
    )

    # Select optimal device
    device = DeviceSelector.get_optimal_device()

    logger.info(f"Starting training on device: {device}")
    # The train() method's DataLoader parameters are not used by the Ultralytics trainer.
    model_trainer.train(None, None, device)


if __name__ == "__main__":
    main()
