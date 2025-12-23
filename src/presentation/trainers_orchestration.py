from pathlib import Path

import torch.optim
from loguru import logger

from src.config import PROJECT_ROOT
from src.infrastructure.dataset_preparer_impl import SklearnDatasetPreparer
from src.infrastructure.trainers import YoloUltralyticsTrainer


def main():
    base_dataset_dir = Path(PROJECT_ROOT) / "datasets"
    source_images = base_dataset_dir / "images"
    source_labels = base_dataset_dir / "labels"

    output_images = Path(PROJECT_ROOT) / "datasets" / "ultralytics" / "images"
    output_labels = Path(PROJECT_ROOT) / "datasets" / "ultralytics" / "labels"

    dataset_preparer = SklearnDatasetPreparer()
    dataset_preparer.prepare_ultralytics_dataset(
        source_images=source_images,
        source_labels=source_labels,
        output_images=output_images,
        output_labels=output_labels,
        train_ratio=0.8,
        val_ratio=0.1,
        move_files=True,
    )
    logger.info("Dataset is prepared and ready for training.")

    # Get data.yaml from project root
    data_config = Path(PROJECT_ROOT / "data.yaml")

    model_weights = "yolov8n.pt"

    epochs = 20
    img_size = 640

    model_trainer = YoloUltralyticsTrainer(model_weights, data_config, epochs, img_size)

    # For Apple M2, choose a device (this is optional since Ultralytics does its own device handling).
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logger.info(f"Starting training on device: {device}")
    # The train() method’s DataLoader parameters are not used by the Ultralytics trainer.
    model_trainer.train(None, None, device)


if __name__ == "__main__":
    main()
