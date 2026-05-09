"""Abstract interface for object detection models.

This module defines the base interface that all object detection models must
implement to enable seamless model swapping in the application.
"""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class Detection(dict[str, Any]):
    """Detection result containing bounding box, class, and confidence.

    Expected format::

        {
            'bbox': [x1, y1, x2, y2],  # Absolute pixel coordinates
            'class_id': int,            # Numeric class identifier
            'class_name': str,          # Human-readable class name
            'confidence': float,        # Detection confidence [0, 1]
        }
    """

    pass


class ObjectDetector(ABC):
    """Abstract base class for object detection models.

    All detector implementations (YOLO, DETR, Faster R-CNN) must inherit
    from this class and implement the predict() and annotate_image() methods.
    """

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
    ) -> list[Detection]:
        """Run inference on image and return detections.

        :param image: Input PIL image
        :param confidence_threshold: Minimum confidence for detections [0, 1]
        :return: List of detections with bbox, class_id, class_name, confidence
        :rtype: List[Detection]
        """
        pass

    @abstractmethod
    def annotate_image(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
    ) -> Image.Image:
        """Annotate image with bounding boxes and labels.

        :param image: Input PIL image
        :param confidence_threshold: Minimum confidence for detections [0, 1]
        :return: Image with bounding boxes and labels drawn
        :rtype: Image.Image
        """
        pass
