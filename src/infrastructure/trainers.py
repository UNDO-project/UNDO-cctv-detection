from pathlib import Path

import torch
from torch import optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from ultralytics import YOLO
from loguru import logger

from src.config import BATCH_SIZE
from src.domain.services.model_trainer import ModelTrainer


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


class FasterRCNNTrainer(ModelTrainer):
    def __init__(
        self, num_classes: int, epochs: int = 10, learning_rate: float = 0.005
    ):
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
        """
        Build and return the Faster R-CNN model with a custom head.
        :return: Faster R-CNN model
        """
        # Load a pre-trained Faster R-CNN model
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one tailored for our dataset
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def train(self, train_loader, val_loader, device) -> None:
        """
        Train the Faster R-CNN model using the provided dataloaders on the specified device.
        :param train_loader: Training dataloader
        :param val_loader: Validation dataloader
        :param device: Device to train on ('cpu', 'cuda', or 'mps')
        :return:
        """
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

    def evaluate(self, val_loader, device) -> None:
        """
        Evaluate the model on the validation set and log the loss.
        :param val_loader: Validation dataloader
        :param device: Device for evaluation
        :return:
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                loss = torch.stack(list(loss_dict.values())).sum()
                total_loss += loss.item()
            logger.info(f"Validation Loss: {total_loss / len(val_loader):.4f}")
