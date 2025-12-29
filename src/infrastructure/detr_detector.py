"""DETR detector wrapper for unified inference interface.

This module provides a DETR (DEtection TRansformer) implementation of the
ObjectDetector interface using HuggingFace Transformers.
"""

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.domain.services.object_detector import Detection, ObjectDetector


class DETRDetector(ObjectDetector):
    """DETR model detector using Hugging Face Transformers.

    Wraps the HuggingFace DETR model to provide a unified interface
    for object detection alongside YOLO and Faster R-CNN models.
    """

    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        device: torch.device | None = None,
    ):
        """Initialize DETR detector.

        :param model_path: Path to DETR model checkpoint directory
        :param class_names: List of class names in order of class IDs
        :param device: Device for inference (cpu, cuda, mps). Defaults to CPU if None.
        """
        self.class_names = class_names
        self.device = device if device is not None else torch.device("cpu")

        # Load model and processor
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.7,
    ) -> list[Detection]:
        """Run DETR inference and return detections.

        DETR typically requires higher confidence thresholds (0.7+) compared
        to YOLO/Faster R-CNN due to its different prediction mechanism.

        :param image: Input PIL image
        :param confidence_threshold: Minimum confidence for detections [0, 1]
        :return: List of detections with bbox, class_id, class_name, confidence
        :rtype: List[Detection]
        """
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold,
        )[0]

        # Format detections
        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"], strict=True
        ):
            detections.append(
                {
                    "bbox": box.tolist(),  # [x1, y1, x2, y2]
                    "class_id": int(label),
                    "class_name": self.class_names[int(label)],
                    "confidence": float(score),
                }
            )

        return detections

    def annotate_image(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.7,
    ) -> Image.Image:
        """Annotate image with DETR predictions.

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
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw label background
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((x1, y1), label, fill="white", font=font)

        return annotated
