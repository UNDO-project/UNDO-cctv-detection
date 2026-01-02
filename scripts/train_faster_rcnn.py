"""Train Faster R-CNN model on CCTV dataset.

This script orchestrates the training of a Faster R-CNN model
on the CCTV detection dataset using PyTorch and torchvision.
"""

import json

import torch
import torchvision.transforms as t
from loguru import logger
from torch.utils.data import DataLoader

from src.config import settings
from src.infrastructure.faster_rcnn_dataset import FasterRCNNDataset
from src.infrastructure.trainers import FasterRCNNTrainer


def collate_fn(batch):
    """Custom collate function for Faster R-CNN batching.

    :param batch: List of (image, target) tuples
    :return: Tuple of (images, targets)
    :rtype: tuple
    """
    return tuple(zip(*batch, strict=True))


def main() -> None:
    """Train Faster R-CNN model on CCTV dataset.

    This function:
    1. Creates Faster R-CNN datasets from YOLO-format annotations
    2. Creates DataLoaders with custom collate function
    3. Initializes Faster R-CNN trainer with configured hyperparameters
    4. Trains the model on the optimal available device
    5. Saves trained weights

    :return: None
    :rtype: None
    """
    logger.info("Starting Faster R-CNN training pipeline...")

    # Define transforms (only ToTensor, FasterRCNN handles normalization)
    transforms = t.Compose(
        [
            t.ToTensor(),
        ]
    )

    # Create datasets from YOLO format
    logger.info("Creating Faster R-CNN datasets...")
    train_dataset = FasterRCNNDataset(
        images_dir=settings.paths.ultralytics_dir / "images" / "train",
        labels_dir=settings.paths.ultralytics_dir / "labels" / "train",
        transforms=transforms,
    )

    val_dataset = FasterRCNNDataset(
        images_dir=settings.paths.ultralytics_dir / "images" / "val",
        labels_dir=settings.paths.ultralytics_dir / "labels" / "val",
        transforms=transforms,
    )

    logger.info(
        f"Prepared {len(train_dataset)} training and "
        f"{len(val_dataset)} validation samples"
    )

    # Create dataloaders with custom collate function
    logger.info("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Initialize trainer
    # num_classes = 3 (background class 0 + CCTV class 1 + CCTV-SIGNS class 2)
    logger.info("Initializing Faster R-CNN trainer...")
    trainer = FasterRCNNTrainer(
        num_classes=3,  # background + 2 CCTV classes
        epochs=settings.training.epochs,
        learning_rate=settings.training.learning_rate,
    )

    # Select device for training
    # Faster R-CNN has compatibility issues with MPS, so prefer CUDA or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Training on CUDA GPU")
    else:
        # Skip MPS due to compatibility issues with Faster R-CNN operations
        device = torch.device("cpu")
        if torch.backends.mps.is_available():
            logger.warning(
                "MPS available but using CPU for Faster R-CNN due to compatibility issues"
            )
        else:
            logger.info("Training on CPU")

    # Train the model
    logger.info("Starting model training...")
    training_metrics = trainer.train(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Save model weights
    logger.info("Saving model weights...")
    settings.models.faster_rcnn_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.model.state_dict(), settings.models.faster_rcnn_weights)

    # Compute mAP metrics on validation set
    logger.info("Computing mAP metrics on validation set...")
    map_metrics = trainer.evaluate_map(val_loader, device)

    # Save training metrics for dashboard
    runs_dir = settings.paths.project_root / "runs" / "faster_rcnn" / "train"
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = runs_dir / "training_metrics.json"

    # Add mAP metrics to training metrics
    training_metrics["final_map50"] = map_metrics["map50"]
    training_metrics["final_map"] = map_metrics["map"]

    with open(metrics_file, "w") as f:
        json.dump(training_metrics, f, indent=2)

    logger.info(f"Training metrics saved to {metrics_file}")
    logger.success(
        f"Faster R-CNN training complete! "
        f"mAP@0.5={map_metrics['map50']:.3f}, mAP@0.5:0.95={map_metrics['map']:.3f}"
    )
    logger.success(f"Model weights saved to {settings.models.faster_rcnn_weights}")


if __name__ == "__main__":
    main()
