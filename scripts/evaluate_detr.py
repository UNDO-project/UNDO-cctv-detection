"""Evaluate DETR model on validation set and save mAP metrics.

This script evaluates a trained DETR model on the validation set and computes
mAP@0.5 and mAP@0.5:0.95 metrics using COCO evaluation protocol. The results
are saved to evaluation_metrics.json for display in the performance dashboard.
"""

import json

from loguru import logger
from torch.utils.data import DataLoader

from src.config import settings
from src.infrastructure.detr_data_preparer import DETRDataPreparer
from src.infrastructure.detr_dataset import DETRDataset
from src.infrastructure.detr_trainer import DETRTrainer
from src.infrastructure.device_selector import DeviceSelector


def main() -> None:
    """Evaluate trained DETR model and save mAP metrics.

    This function:
    1. Loads the trained DETR model from runs/detr/final
    2. Prepares validation dataset
    3. Runs inference and computes COCO mAP metrics
    4. Saves evaluation results to evaluation_metrics.json

    :return: None
    :rtype: None
    """
    logger.info("Starting DETR evaluation...")

    # Create trainer and load existing model
    logger.info("Loading DETR model...")
    trainer = DETRTrainer(
        model_name=settings.models.detr_model_name,
        num_labels=2,
        epochs=settings.training.epochs,
        batch_size=settings.training.batch_size,
    )

    # Check if trained model exists
    model_path = settings.paths.project_root / "runs" / "detr" / "final"
    if not model_path.exists():
        logger.error(
            f"No trained DETR model found at {model_path}. "
            "Please train DETR first using 'uv run cctv-train-detr'"
        )
        return

    # Prepare validation data
    logger.info("Preparing validation dataset...")
    preparer = DETRDataPreparer(class_names=["CCTV", "CCTV-SIGNS"])
    val_annotations = preparer.prepare_annotations(
        images_dir=settings.paths.ultralytics_dir / "images" / "val",
        labels_dir=settings.paths.ultralytics_dir / "labels" / "val",
    )

    logger.info(f"Prepared {len(val_annotations)} validation samples")

    # Create validation dataset
    val_dataset = DETRDataset(
        image_paths=[ann["image_path"] for ann in val_annotations],
        annotations=val_annotations,
        processor=trainer.processor,
    )

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        collate_fn=trainer.collate_fn,
    )

    # Select device
    device = DeviceSelector.get_optimal_device()
    logger.info(f"Running evaluation on {device}")

    # Evaluate model
    metrics = trainer.evaluate_map(val_loader, device)

    # Save metrics
    eval_file = model_path / "evaluation_metrics.json"
    eval_file.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success(f"Evaluation complete! Metrics saved to {eval_file}")
    logger.info(
        f"Results: mAP@0.5 = {metrics['map50']:.3f}, "
        f"mAP@0.5:0.95 = {metrics['map']:.3f}"
    )


if __name__ == "__main__":
    main()
