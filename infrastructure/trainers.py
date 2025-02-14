from pathlib import Path

from ultralytics import YOLO
from loguru import logger

from config import BATCH_SIZE
from domain.services.model_trainer import ModelTrainer


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
