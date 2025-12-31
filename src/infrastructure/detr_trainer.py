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

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DETR training.

        Handles variable-sized images by padding to the largest dimensions
        in the batch, and keeps labels as a list (variable number of boxes).

        :param batch: List of dicts with 'pixel_values' and 'labels'
        :return: Batched dict with padded pixel_values, pixel_mask, and labels
        :rtype: dict
        """
        # Get pixel values and find max dimensions
        pixel_values_list = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Find max height and width in batch
        max_height = max([pv.shape[1] for pv in pixel_values_list])
        max_width = max([pv.shape[2] for pv in pixel_values_list])

        # Pad all images to max dimensions
        padded_pixel_values = []
        pixel_mask = []

        for pv in pixel_values_list:
            c, h, w = pv.shape
            # Create padded tensor (pad with zeros)
            padded = torch.zeros((c, max_height, max_width), dtype=pv.dtype)
            padded[:, :h, :w] = pv

            # Create mask (1 for real pixels, 0 for padding)
            mask = torch.zeros((max_height, max_width), dtype=torch.long)
            mask[:h, :w] = 1

            padded_pixel_values.append(padded)
            pixel_mask.append(mask)

        return {
            "pixel_values": torch.stack(padded_pixel_values),
            "pixel_mask": torch.stack(pixel_mask),
            "labels": labels,
        }

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
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            remove_unused_columns=False,  # Keep all columns
            push_to_hub=False,
        )

        # Create Hugging Face Trainer with custom collate function
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            tokenizer=self.processor,  # Used for saving
            data_collator=self.collate_fn,  # Custom collator for variable-sized labels
        )

        # Train
        logger.info("Starting DETR training...")
        trainer.train()

        # Save final model
        final_path = self.output_dir / "final"
        trainer.save_model(str(final_path))
        logger.success(f"DETR training complete. Model saved to {final_path}")

    def evaluate_map(
        self, val_loader: DataLoader, device: torch.device
    ) -> dict[str, float]:
        """Evaluate mAP on validation set using COCO metrics.

        Runs inference on validation set and computes mAP@0.5 and mAP@0.5:0.95
        compatible with YOLO metrics for fair comparison.

        :param val_loader: Validation dataloader
        :param device: Device for inference
        :return: Dictionary with map50 and map (mAP@0.5:0.95)
        :rtype: dict[str, float]
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        logger.info("Evaluating DETR mAP on validation set...")

        self.model.to(device)
        self.model.eval()

        # Collect predictions and ground truth in COCO format
        coco_gt = {"images": [], "annotations": [], "categories": []}
        coco_predictions = []

        # Add categories (CCTV classes)
        for i in range(self.num_labels):
            coco_gt["categories"].append({"id": i, "name": f"class_{i}"})

        annotation_id = 0
        image_id = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = batch["labels"]

                # Run inference
                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                # Process each image in batch
                for i in range(len(labels)):
                    # Get image size from mask
                    h, w = (
                        pixel_mask[i].sum(dim=0).max().item(),
                        pixel_mask[i].sum(dim=1).max().item(),
                    )

                    # Add image to ground truth
                    coco_gt["images"].append(
                        {"id": image_id, "width": int(w), "height": int(h)}
                    )

                    # Add ground truth annotations
                    gt_boxes = labels[i]["boxes"]
                    gt_labels = labels[i]["class_labels"]

                    for box, label in zip(gt_boxes, gt_labels, strict=False):
                        # Convert from center format to COCO format (x, y, w, h)
                        cx, cy, box_w, box_h = box.tolist()
                        x = (cx - box_w / 2) * w
                        y = (cy - box_h / 2) * h
                        box_w_abs = box_w * w
                        box_h_abs = box_h * h

                        coco_gt["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": int(label),
                                "bbox": [x, y, box_w_abs, box_h_abs],
                                "area": box_w_abs * box_h_abs,
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1

                    # Process predictions
                    logits = outputs.logits[i]
                    pred_boxes = outputs.pred_boxes[i]

                    # Get scores and labels
                    probs = logits.softmax(-1)
                    scores, pred_labels = probs[..., :-1].max(-1)

                    # Convert boxes from normalized center to COCO format
                    for score, label, box in zip(
                        scores, pred_labels, pred_boxes, strict=False
                    ):
                        if score > 0.05:  # Threshold for evaluation
                            cx, cy, box_w, box_h = box.tolist()
                            x = (cx - box_w / 2) * w
                            y = (cy - box_h / 2) * h
                            box_w_abs = box_w * w
                            box_h_abs = box_h * h

                            coco_predictions.append(
                                {
                                    "image_id": image_id,
                                    "category_id": int(label),
                                    "bbox": [x, y, box_w_abs, box_h_abs],
                                    "score": float(score),
                                }
                            )

                    image_id += 1

        # Create COCO objects and evaluate
        import json
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
        import os

        os.unlink(gt_file)

        logger.success(
            f"DETR Evaluation: mAP@0.5={map50:.3f}, mAP@0.5:0.95={map_full:.3f}"
        )

        return {"map50": float(map50), "map": float(map_full)}

    def save_weights(self, path: Path) -> None:
        """Save model weights in PyTorch format.

        :param path: Path to save weights (e.g., samples/detr_best.pt)
        :return: None
        :rtype: None
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"DETR weights saved to {path}")
