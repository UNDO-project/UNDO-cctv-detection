"""Train DETR model on CCTV dataset.

This script orchestrates the training of a DETR (DEtection TRansformer) model
on the CCTV detection dataset. It handles data preparation, dataset creation,
and model training using the HuggingFace Transformers library.
"""

from loguru import logger
from torch.utils.data import DataLoader

from src.config import settings
from src.infrastructure.detr_data_preparer import DETRDataPreparer
from src.infrastructure.detr_dataset import DETRDataset
from src.infrastructure.detr_trainer import DETRTrainer
from src.infrastructure.device_selector import DeviceSelector


def main():
    """Train DETR model on CCTV dataset.

    This function:
    1. Prepares COCO-format annotations from YOLO labels
    2. Creates DETR datasets for training and validation
    3. Initializes DETR trainer with configured hyperparameters
    4. Trains the model on the optimal available device
    5. Saves trained weights
    """
    logger.info("Starting DETR training pipeline...")

    # Prepare annotations from YOLO format
    logger.info("Preparing DETR dataset from YOLO annotations...")
    preparer = DETRDataPreparer(class_names=["CCTV", "CCTV-SIGNS"])

    train_annotations = preparer.prepare_annotations(
        images_dir=settings.paths.ultralytics_dir / "images" / "train",
        labels_dir=settings.paths.ultralytics_dir / "labels" / "train",
    )

    val_annotations = preparer.prepare_annotations(
        images_dir=settings.paths.ultralytics_dir / "images" / "val",
        labels_dir=settings.paths.ultralytics_dir / "labels" / "val",
    )

    logger.info(
        f"Prepared {len(train_annotations)} training and "
        f"{len(val_annotations)} validation samples"
    )

    # Create trainer and initialize processor
    logger.info("Initializing DETR trainer...")
    trainer = DETRTrainer(
        model_name=settings.models.detr_model_name,
        num_labels=2,
        epochs=settings.training.epochs,
        batch_size=settings.training.batch_size,
    )

    # Create datasets using the trainer's processor
    logger.info("Creating DETR datasets...")
    train_dataset = DETRDataset(
        image_paths=[ann["image_path"] for ann in train_annotations],
        annotations=train_annotations,
        processor=trainer.processor,
    )

    val_dataset = DETRDataset(
        image_paths=[ann["image_path"] for ann in val_annotations],
        annotations=val_annotations,
        processor=trainer.processor,
    )

    # Create dataloaders with DETR's custom collate function
    # This handles variable-sized images by padding and creating pixel masks
    logger.info("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,
        collate_fn=trainer.collate_fn,  # Use trainer's collate function
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        collate_fn=trainer.collate_fn,  # Use trainer's collate function
    )

    # Select optimal device for training
    device = DeviceSelector.get_optimal_device()
    logger.info(f"Training on device: {device}")

    # Train the model
    logger.info("Starting model training...")
    trainer.train(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Automatically evaluate mAP after training
    logger.info("Evaluating model performance...")
    eval_metrics = trainer.evaluate_map(val_loader, device)

    # Save evaluation metrics
    import json

    eval_file = trainer.output_dir / "final" / "evaluation_metrics.json"
    with open(eval_file, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.success(
        f"DETR training complete! Model saved to {trainer.output_dir / 'final'}"
    )
    logger.info(
        f"Evaluation: mAP@0.5={eval_metrics['map50']:.3f}, "
        f"mAP@0.5:0.95={eval_metrics['map']:.3f}"
    )


if __name__ == "__main__":
    main()
