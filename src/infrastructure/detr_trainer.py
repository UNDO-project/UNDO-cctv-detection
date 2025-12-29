"""DETR model trainer using Hugging Face Transformers.

This module provides a ModelTrainer implementation for training DETR (DEtection
TRansformer) models on the CCTV detection dataset using the HuggingFace
Transformers library.
"""

from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    Trainer,
    TrainingArguments,
)

from src.config import settings
from src.domain.services.model_trainer import ModelTrainer


class DETRTrainer(ModelTrainer):
    """DETR model trainer using Hugging Face Transformers.

    DETR (DEtection TRansformer) is an end-to-end object detection model using
    transformers instead of CNNs. It eliminates the need for hand-designed
    components like NMS (non-maximum suppression) by using a set-based global
    loss and bipartite matching.
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        num_labels: int = 2,  # CCTV, CCTV-SIGNS
        epochs: int = 20,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        output_dir: Path | None = None,
    ):
        """Initialize DETR trainer.

        :param model_name: Hugging Face model identifier
        :param num_labels: Number of detection classes (excluding background)
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for AdamW optimizer
        :param batch_size: Training batch size
        :param output_dir: Directory to save checkpoints
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_dir = output_dir or (settings.paths.project_root / "runs" / "detr")

        # Initialize processor and model
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,  # Allow head replacement
        )

        logger.info(f"Initialized DETR trainer with {model_name}")

    def train(
        self,
        device: torch.device,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        data_config: Path | None = None,
    ) -> None:
        """Train DETR model using PyTorch DataLoaders.

        :param device: Training device
        :param train_loader: Training dataloader (required)
        :param val_loader: Validation dataloader (required)
        :param data_config: Not used for DETR
        :return: None
        :rtype: None
        :raises ValueError: If train_loader or val_loader not provided
        """
        self.validate_inputs(train_loader, val_loader, data_config)

        if train_loader is None or val_loader is None:
            raise ValueError("DetrTrainer requires train_loader and val_loader")

        logger.info(f"Training DETR on {device}")

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=1e-4,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            remove_unused_columns=False,  # Keep all columns
            push_to_hub=False,
        )

        # Create Hugging Face Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            tokenizer=self.processor,  # Used for saving
        )

        # Train
        logger.info("Starting DETR training...")
        trainer.train()

        # Save final model
        final_path = self.output_dir / "final"
        trainer.save_model(str(final_path))
        logger.success(f"DETR training complete. Model saved to {final_path}")

    def save_weights(self, path: Path) -> None:
        """Save model weights in PyTorch format.

        :param path: Path to save weights (e.g., samples/detr_best.pt)
        :return: None
        :rtype: None
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"DETR weights saved to {path}")
