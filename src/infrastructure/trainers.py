from pathlib import Path

import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
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
    ) -> dict:
        """Train the Faster R-CNN model using the provided dataloaders.

        Faster R-CNN training uses DataLoaders, not config files.

        :param device: Device to train on (cpu, cuda, or mps)
        :param train_loader: Training dataloader (required for Faster R-CNN)
        :param val_loader: Validation dataloader (required for Faster R-CNN)
        :param data_config: Not used by Faster R-CNN
        :return: Training metrics dictionary with losses per epoch
        :rtype: dict
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

        # Track metrics for dashboard
        train_losses = []
        val_losses = []

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

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}], Training Loss: {avg_train_loss:.4f}"
            )
            # Evaluate on the validation set after each epoch
            val_loss = self.evaluate(val_loader, device)
            val_losses.append(val_loss)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epochs": self.epochs,
        }

    def evaluate(self, val_loader: DataLoader, device: torch.device) -> float:
        """Evaluate the model on the validation set and return the loss.

        Note: Model must stay in training mode to return loss dictionaries.
        We use torch.no_grad() to prevent gradient computation.

        :param val_loader: Validation dataloader
        :param device: Device for evaluation
        :return: Average validation loss
        :rtype: float
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
        avg_loss = total_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate_map(
        self,
        val_loader: DataLoader,
        device: torch.device,
        score_threshold: float = 0.05,
    ) -> dict[str, float]:
        """Evaluate model using COCO mAP metrics.

        :param val_loader: Validation dataloader
        :param device: Device for evaluation
        :param score_threshold: Minimum score threshold for predictions
        :return: Dictionary with mAP@0.5 and mAP@0.5:0.95
        :rtype: dict
        """
        logger.info("Computing mAP metrics using COCO evaluation...")
        self.model.eval()
        self.model.to(device)

        # Prepare COCO format data structures
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": f"class_{i}"} for i in range(self.num_classes)
            ],
        }
        coco_predictions = []

        annotation_id = 0
        image_id = 0

        with torch.no_grad():
            for images, targets in val_loader:
                # Move images to device
                images = [img.to(device) for img in images]

                # Run inference
                predictions = self.model(images)

                # Process each image in batch
                for _i, (img, target, pred) in enumerate(
                    zip(images, targets, predictions, strict=False)
                ):
                    # Get image dimensions (C, H, W)
                    _, h, w = img.shape

                    # Add image to ground truth
                    coco_gt["images"].append(
                        {"id": image_id, "width": int(w), "height": int(h)}
                    )

                    # Add ground truth annotations
                    gt_boxes = target["boxes"]  # Format: [x1, y1, x2, y2]
                    gt_labels = target["labels"]

                    for box, label in zip(gt_boxes, gt_labels, strict=False):
                        # Convert from [x1, y1, x2, y2] to COCO format [x, y, w, h]
                        x1, y1, x2, y2 = box.tolist()
                        box_w = x2 - x1
                        box_h = y2 - y1

                        coco_gt["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": int(label),
                                "bbox": [x1, y1, box_w, box_h],
                                "area": box_w * box_h,
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1

                    # Process predictions
                    pred_boxes = pred["boxes"]  # Format: [x1, y1, x2, y2]
                    pred_labels = pred["labels"]
                    pred_scores = pred["scores"]

                    # Filter by score threshold
                    for box, label, score in zip(
                        pred_boxes, pred_labels, pred_scores, strict=False
                    ):
                        if score > score_threshold:
                            # Convert from [x1, y1, x2, y2] to COCO format [x, y, w, h]
                            x1, y1, x2, y2 = box.tolist()
                            box_w = x2 - x1
                            box_h = y2 - y1

                            coco_predictions.append(
                                {
                                    "image_id": image_id,
                                    "category_id": int(label),
                                    "bbox": [x1, y1, box_w, box_h],
                                    "score": float(score),
                                }
                            )

                    image_id += 1

        # Create COCO objects and evaluate
        import json
        import os
        import tempfile

        # Write ground truth to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(coco_gt, f)
            gt_file = f.name

        # Create COCO ground truth object
        coco_gt_obj = COCO(gt_file)

        # Load predictions
        coco_dt = coco_gt_obj.loadRes(coco_predictions) if coco_predictions else None

        if coco_dt is None or len(coco_predictions) == 0:
            logger.warning("No predictions generated, mAP = 0.0")
            os.unlink(gt_file)
            return {"map50": 0.0, "map": 0.0}

        # Evaluate
        coco_eval = COCOeval(coco_gt_obj, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        map50 = coco_eval.stats[1]  # mAP@0.5
        map_full = coco_eval.stats[0]  # mAP@0.5:0.95

        # Clean up temp file
        os.unlink(gt_file)

        logger.success(
            f"Faster R-CNN Evaluation: mAP@0.5={map50:.3f}, mAP@0.5:0.95={map_full:.3f}"
        )

        return {"map50": float(map50), "map": float(map_full)}
