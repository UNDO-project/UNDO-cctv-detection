"""YOLO detector wrapper for unified inference interface.

This module provides a YOLO implementation of the ObjectDetector interface
using the Ultralytics library.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.domain.services.object_detector import Detection, ObjectDetector


class YOLODetector(ObjectDetector):
    """YOLO model detector using Ultralytics library.

    Wraps the Ultralytics YOLO model to provide a unified interface
    for object detection alongside DETR and Faster R-CNN models.
    """

    def __init__(self, model_path: Path, class_names: list[str]):
        """Initialize YOLO detector.

        :param model_path: Path to YOLO weights (.pt file)
        :param class_names: List of class names in order of class IDs
        """
        self.model = YOLO(str(model_path))
        self.class_names = class_names

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
    ) -> list[Detection]:
        """Run YOLO inference and return detections.

        :param image: Input PIL image
        :param confidence_threshold: Minimum confidence for detections [0, 1]
        :return: List of detections with bbox, class_id, class_name, confidence
        :rtype: List[Detection]
        """
        results = self.model.predict(
            source=np.array(image),
            conf=confidence_threshold,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            detections.append(
                {
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "class_id": int(box.cls[0]),
                    "class_name": self.class_names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                }
            )

        return detections

    def annotate_image(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
    ) -> Image.Image:
        """Annotate image with YOLO predictions.

        :param image: Input PIL image
        :param confidence_threshold: Minimum confidence for detections [0, 1]
        :return: Image with bounding boxes and labels drawn
        :rtype: Image.Image
        """
        results = self.model.predict(
            source=np.array(image),
            conf=confidence_threshold,
            verbose=False,
        )
        annotated = results[0].plot()
        return Image.fromarray(annotated)
