from pathlib import Path

import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from ultralytics import YOLO

from src.config import settings
from src.domain.services.model_trainer import ModelTrainer


class YoloUltralyticsTrainer(ModelTrainer):
    def __init__(
        self, model_weights: str, data_config: Path, epochs: int, img_size: int = 640
    ) -> None:
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

    def train(
        self,
        device: torch.device,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        data_config: Path | None = None,
    ) -> None:
        """Train the YOLO model using Ultralytics library.

        YOLO training uses the data_config YAML file, not DataLoaders.

        :param device: Target device for training
        :param train_loader: Not used by YOLO (uses data_config instead)
        :param val_loader: Not used by YOLO (uses data_config instead)
        :param data_config: Path to YAML config file (required for YOLO)
        :return: None
        :rtype: None
        :raises ValueError: If data_config is not provided
        """
        # Validate that data_config is provided
        config_to_use = data_config or self.data_config
        self.validate_inputs(train_loader, val_loader, config_to_use)

        if config_to_use is None:
            msg = "YOLO trainer requires data_config to be provided"
            raise ValueError(msg)

        # Note: Ultralytics handles device selection internally
        logger.info(f"Training with Ultralytics YOLO on device: {device}")
        self.model.train(
            data=config_to_use,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=settings.training.batch_size,
            lr0=settings.training.learning_rate,
            weight_decay=0.001,  # Regularization to mitigate overfitting
            mosaic=0.8,  # Mosaic augmentation
            mixup=0.1,  # Mixup augmentation
            freeze=[
                0,
                1,
                2,
                3,
            ],  # Freeze early layers to leverage pretrained features
        )


class FasterRCNNTrainer(ModelTrainer):
    def __init__(
        self, num_classes: int, epochs: int = 10, learning_rate: float = 0.005
    ) -> None:
        """
        Initialize the Faster R-CNN trainer with model and training parameters.
        :param num_classes: Number of classes (including background)
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for the optimizer
        """
        self.num_classes = num_classes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self) -> FasterRCNN:
        """Build and return the Faster R-CNN model with a custom head.

        :return: Faster R-CNN model
        :rtype: FasterRCNN
        """
        # Load a pre-trained Faster R-CNN model with COCO weights
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one tailored for our dataset
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def train(
        self,
        device: torch.device,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        data_config: Path | None = None,
    ) -> None:
        """Train the Faster R-CNN model using the provided dataloaders.

        Faster R-CNN training uses DataLoaders, not config files.

        :param device: Device to train on (cpu, cuda, or mps)
        :param train_loader: Training dataloader (required for Faster R-CNN)
        :param val_loader: Validation dataloader (required for Faster R-CNN)
        :param data_config: Not used by Faster R-CNN
        :return: None
        :rtype: None
        :raises ValueError: If train_loader or val_loader not provided
        """
        # Validate that loaders are provided
        self.validate_inputs(train_loader, val_loader, data_config)

        if train_loader is None or val_loader is None:
            msg = "Faster R-CNN trainer requires train_loader and val_loader"
            raise ValueError(msg)

        self.model.to(device)
        # Optimizer for parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(
            params=params, lr=self.learning_rate, momentum=0.9, weight_decay=0.0005
        )

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for images, targets in train_loader:
                # Move images and targets to the specified device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                # Forward pass returns a dict of losses
                loss_dict = self.model(images, targets)
                loss = torch.stack(list(loss_dict.values())).sum()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}], Training Loss: {epoch_loss / len(train_loader):.4f}"
            )
            # Optionally, evaluate on the validation set after each epoch
            self.evaluate(val_loader, device)

    def evaluate(self, val_loader: DataLoader, device: torch.device) -> None:
        """
        Evaluate the model on the validation set and log the loss.

        Note: Model must stay in training mode to return loss dictionaries.
        We use torch.no_grad() to prevent gradient computation.

        :param val_loader: Validation dataloader
        :param device: Device for evaluation
        :return:
        """
        self.model.train()  # Keep in train mode to get losses
        total_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                loss = torch.stack(list(loss_dict.values())).sum()
                total_loss += loss.item()
            logger.info(f"Validation Loss: {total_loss / len(val_loader):.4f}")
