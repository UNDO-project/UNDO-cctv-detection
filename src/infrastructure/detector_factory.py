"""Factory for creating object detection model instances.

This module provides a factory pattern implementation for instantiating
different object detection models with a unified interface.
"""

from pathlib import Path
from typing import Literal

import torch

from src.domain.services.object_detector import ObjectDetector
from src.infrastructure.detr_detector import DETRDetector
from src.infrastructure.faster_rcnn_detector import FasterRCNNDetector
from src.infrastructure.yolo_detector import YOLODetector

ModelType = Literal["yolo", "detr", "faster-rcnn"]


class DetectorFactory:
    """Factory for creating model detectors with unified interface.

    Provides a centralized way to instantiate different detection models
    (YOLO, DETR, Faster R-CNN) with consistent parameters.
    """

    @staticmethod
    def create_detector(
        model_type: ModelType,
        model_path: Path,
        class_names: list[str],
        device: torch.device | None = None,
    ) -> ObjectDetector:
        """Create detector for specified model type.

        :param model_type: Type of model ("yolo", "detr", "faster-rcnn")
        :param model_path: Path to model weights
        :param class_names: List of class names in order of class IDs
        :param device: Device for inference (cpu, cuda, mps). Defaults to CPU if None.
        :return: ObjectDetector instance configured for the specified model
        :rtype: ObjectDetector
        :raises ValueError: If model_type is not recognized
        """
        if device is None:
            device = torch.device("cpu")

        if model_type == "yolo":
            return YOLODetector(model_path, class_names)
        elif model_type == "detr":
            return DETRDetector(model_path, class_names, device)
        elif model_type == "faster-rcnn":
            return FasterRCNNDetector(model_path, class_names, device)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: 'yolo', 'detr', 'faster-rcnn'"
            )
