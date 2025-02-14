from pathlib import Path

import torch
from ultralytics import YOLO
from loguru import logger

from config import BATCH_SIZE
from domain.services.model_trainer import ModelTrainer


class YoloTrainer(ModelTrainer):
    def __init__(self, model, optimizer, num_epochs):
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self, train_loader, val_loader, device):
        self.model.to(device)
        logger.info(f"Training on device: {device}")
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for images, targets in train_loader:
                # Move the batch to the device:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                # Use tensor operations to get the sum:
                losses = torch.stack(list(loss_dict.values())).sum()
                total_loss += losses.item()

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} completed, loss: {total_loss:.4f}"
            )


class YoloUltralyticsTrainer(ModelTrainer):
    def __init__(
        self, model_weights: str, data_config: Path, epochs: int, img_size: int = 640
    ):
        """
        :param model_weights: Path to a YOLO model weight file or a model name (e.g., 'yolov8n.pt').
        :param data_config: Path to a YAML file with the data configuration for training.
        :param epochs: Number of training epochs.
        :param img_size: Image size to use for training.
        """
        self.model_weights = model_weights
        self.data_config = data_config
        self.epochs = epochs
        self.img_size = img_size
        # Initialize YOLO model from ultralytics
        self.model = YOLO(model_weights)

    def train(self, train_loader, val_loader, device):
        # Note: Ultralytics handles device selection internally (and often chooses the best available one).
        logger.info(f"Training with Ultralytics YOLO on device: {device}")
        self.model.train(
            data=self.data_config,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=BATCH_SIZE,  # Lower batch size for a small dataset
            lr0=0.005,  # Reduced initial learning rate
            weight_decay=0.001,  # Increased regularization to mitigate overfitting
            mosaic=0.8,  # Slightly reduced mosaic augmentation
            mixup=0.1,  # Apply a small mixup augmentation factor
            freeze=[
                0,
                1,
                2,
                3,
            ],  # Freeze the early layers to leverage pretrained features
        )
