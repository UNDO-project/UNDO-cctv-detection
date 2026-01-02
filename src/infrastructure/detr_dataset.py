"""DETR-compatible dataset wrapper for CCTV detection training.

This module provides a PyTorch Dataset implementation that converts COCO-format
annotations to DETR-compatible format for transformer-based object detection.
"""

from typing import Any

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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get image and target for training.

        :param idx: Index of sample to retrieve
        :return: Dictionary with:
            - pixel_values: Preprocessed image tensor
            - labels: Target dict containing class_labels, boxes, image_id, area, iscrowd
        :rtype: dict[str, Any]
        """
        image_path = self.image_paths[idx]
        ann = self.annotations[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Convert to COCO format expected by DETR processor
        # The processor expects: {"image_id": int, "annotations": [{"bbox": [...], "category_id": int, ...}, ...]}
        # For negative samples (images with no objects), annotations will be an empty list
        coco_annotations = []
        for i in range(len(ann["boxes"])):
            coco_annotations.append(
                {
                    "bbox": ann["boxes"][i],
                    "category_id": ann["category_ids"][i],
                    "area": ann["area"][i],
                    "iscrowd": ann["iscrowd"][i],
                }
            )

        target = {
            "image_id": ann["image_id"],
            "annotations": coco_annotations,  # Empty list for negative samples
        }

        # Preprocess
        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        # Remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze(0)

        # Extract and flatten the labels to regular tensors
        labels = encoding["labels"][0]

        # Convert BatchFeature to regular dict with tensors
        labels_dict = {
            "class_labels": labels["class_labels"],
            "boxes": labels["boxes"],
        }

        return {
            "pixel_values": pixel_values,
            "labels": labels_dict,
        }
