"""DETR-compatible dataset wrapper for CCTV detection training.

This module provides a PyTorch Dataset implementation that converts COCO-format
annotations to DETR-compatible format for transformer-based object detection.
"""

from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import DetrImageProcessor


class DETRDataset(Dataset):
    """Dataset wrapper for DETR training.

    Converts COCO-format annotations to DETR format with proper preprocessing
    using the HuggingFace DetrImageProcessor.
    """

    def __init__(
        self,
        image_paths: list[str],
        annotations: list[dict[str, Any]],
        processor: DetrImageProcessor,
        max_size: int = 800,
    ):
        """Initialize DETR dataset.

        :param image_paths: List of image file paths
        :param annotations: List of COCO-format annotations
        :param processor: DETR image processor for preprocessing
        :param max_size: Maximum image size for resizing
        """
        self.image_paths = image_paths
        self.annotations = annotations
        self.processor = processor
        self.max_size = max_size

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        :return: Number of images in dataset
        :rtype: int
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get image and target for training.

        :param idx: Index of sample to retrieve
        :return: Tuple of (pixel_values, target_dict) where target_dict contains:
            - class_labels: [num_objects] class IDs
            - boxes: [num_objects, 4] bounding boxes (cx, cy, w, h normalized)
            - image_id: Image identifier
            - area: Area of each bounding box
            - iscrowd: Whether each object is a crowd annotation
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        """
        image_path = self.image_paths[idx]
        ann = self.annotations[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Format target
        target = {
            "class_labels": torch.tensor(ann["category_ids"], dtype=torch.long),
            "boxes": torch.tensor(ann["boxes"], dtype=torch.float32),
            "image_id": torch.tensor([ann["image_id"]]),
            "area": torch.tensor(ann["area"], dtype=torch.float32),
            "iscrowd": torch.tensor(ann["iscrowd"], dtype=torch.int64),
        }

        # Preprocess
        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        # Remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze(0)
        target = encoding["labels"][0]

        return pixel_values, target
