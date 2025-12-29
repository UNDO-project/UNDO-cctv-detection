"""Faster R-CNN detector wrapper for unified inference interface.

This module provides a Faster R-CNN implementation of the ObjectDetector interface
using PyTorch and torchvision.
"""

from pathlib import Path

import torch
import torchvision.transforms as t
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.domain.services.object_detector import Detection, ObjectDetector


class FasterRCNNDetector(ObjectDetector):
    """Faster R-CNN model detector using PyTorch and torchvision.

    Wraps the torchvision Faster R-CNN model to provide a unified interface
    for object detection alongside YOLO and DETR models.
    """

    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        device: torch.device | None = None,
    ):
        """Initialize Faster R-CNN detector.

        :param model_path: Path to Faster R-CNN weights (.pt file)
        :param class_names: List of class names in order of class IDs
        :param device: Device for inference (cpu, cuda, mps). Defaults to CPU if None.
        """
        self.class_names = class_names
        self.device = device if device is not None else torch.device("cpu")
        self.num_classes = len(class_names) + 1  # +1 for background

        # Build model architecture
        self.model = self._build_model()

        # Load trained weights
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Image transform
        self.transform = t.ToTensor()

    def _build_model(self):
        """Build Faster R-CNN model with custom head.

        :return: Faster R-CNN model
        """
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
    ) -> list[Detection]:
        """Run Faster R-CNN inference and return detections.

        :param image: Input PIL image
        :param confidence_threshold: Minimum confidence for detections [0, 1]
        :return: List of detections with bbox, class_id, class_name, confidence
        :rtype: List[Detection]
        """
        # Preprocess image
        img_tensor = self.transform(image).to(self.device)

        # Inference
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]

        # Filter by confidence and format detections
        detections = []
        for box, label, score in zip(
            predictions["boxes"],
            predictions["labels"],
            predictions["scores"],
            strict=True,
        ):
            if score >= confidence_threshold:
                # Label 0 is background, so subtract 1 for class_id
                class_id = int(label) - 1
                if 0 <= class_id < len(self.class_names):
                    detections.append(
                        {
                            "bbox": box.cpu().tolist(),  # [x1, y1, x2, y2]
                            "class_id": class_id,
                            "class_name": self.class_names[class_id],
                            "confidence": float(score),
                        }
                    )

        return detections

    def annotate_image(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
    ) -> Image.Image:
        """Annotate image with Faster R-CNN predictions.

        :param image: Input PIL image
        :param confidence_threshold: Minimum confidence for detections [0, 1]
        :return: Image with bounding boxes and labels drawn
        :rtype: Image.Image
        """
        detections = self.predict(image, confidence_threshold)

        # Draw on copy
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # Try to load font, fallback to default
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']}: {det['confidence']:.2f}"

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

            # Draw label background
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_bbox, fill="green")
            draw.text((x1, y1), label, fill="white", font=font)

        return annotated
