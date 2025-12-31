"""Evaluate Faster R-CNN model on validation set and save mAP metrics.

This script evaluates a trained Faster R-CNN model on the validation set and computes
mAP@0.5 and mAP@0.5:0.95 metrics using COCO evaluation protocol. The results
are saved to evaluation_metrics.json for display in the performance dashboard.
"""

import json

import torch
import torchvision.transforms as t
from loguru import logger
from torch.utils.data import DataLoader

from src.config import settings
from src.infrastructure.faster_rcnn_dataset import FasterRCNNDataset
from src.infrastructure.trainers import FasterRCNNTrainer


def main() -> None:
    """Evaluate trained Faster R-CNN model and save mAP metrics.

    This function:
    1. Loads the trained Faster R-CNN model
    2. Prepares validation dataset
    3. Runs inference and computes COCO mAP metrics
    4. Saves evaluation results to evaluation_metrics.json

    :return: None
    :rtype: None
    """
    logger.info("Starting Faster R-CNN evaluation...")

    # Check if trained model exists
    model_path = settings.models.faster_rcnn_weights
    if not model_path.exists():
        logger.error(
            f"No trained Faster R-CNN model found at {model_path}. "
            "Please train Faster R-CNN first using 'uv run cctv-train-faster-rcnn'"
        )
        return

    # Create trainer and load model weights
    logger.info("Loading Faster R-CNN model...")
    trainer = FasterRCNNTrainer(
        num_classes=3,  # Background + 2 classes (CCTV, CCTV-SIGNS)
        epochs=settings.training.epochs,
        learning_rate=settings.training.learning_rate,
    )

    # Load trained weights
    trainer.model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info(f"Loaded weights from {model_path}")

    # Prepare validation dataset
    logger.info("Preparing validation dataset...")
    transform = t.Compose([t.ToTensor()])

    # Load validation dataset directly (already split during data preparation)
    val_dataset = FasterRCNNDataset(
        images_dir=settings.paths.ultralytics_dir / "images" / "val",
        labels_dir=settings.paths.ultralytics_dir / "labels" / "val",
        transforms=transform,
    )

    logger.info(f"Prepared {len(val_dataset)} validation samples")

    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x, strict=True)),
    )

    # Select device for evaluation
    # Faster R-CNN has compatibility issues with MPS, so prefer CUDA or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Running evaluation on CUDA GPU")
    else:
        # Skip MPS due to compatibility issues with Faster R-CNN operations
        device = torch.device("cpu")
        if torch.backends.mps.is_available():
            logger.warning(
                "MPS available but using CPU for Faster R-CNN due to compatibility issues"
            )
        else:
            logger.info("Running evaluation on CPU")

    # Evaluate model
    metrics = trainer.evaluate_map(val_loader, device)

    # Save metrics
    runs_dir = settings.paths.project_root / "runs" / "faster_rcnn" / "train"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Save standalone evaluation metrics
    eval_file = runs_dir / "evaluation_metrics.json"
    with open(eval_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Update training_metrics.json if it exists
    training_metrics_file = runs_dir / "training_metrics.json"
    if training_metrics_file.exists():
        logger.info("Updating training_metrics.json with new mAP values...")
        with open(training_metrics_file) as f:
            training_metrics = json.load(f)

        # Update mAP values
        training_metrics["final_map50"] = metrics["map50"]
        training_metrics["final_map"] = metrics["map"]

        with open(training_metrics_file, "w") as f:
            json.dump(training_metrics, f, indent=2)

        logger.info(f"Updated {training_metrics_file}")
    else:
        logger.warning(
            f"training_metrics.json not found at {training_metrics_file}. "
            "Only evaluation_metrics.json was saved."
        )

    logger.success(f"Evaluation complete! Metrics saved to {eval_file}")
    logger.info(
        f"Results: mAP@0.5 = {metrics['map50']:.3f}, "
        f"mAP@0.5:0.95 = {metrics['map']:.3f}"
    )


if __name__ == "__main__":
    main()
