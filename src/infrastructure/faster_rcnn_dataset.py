"""Faster R-CNN dataset wrapper for CCTV detection training.

This module provides a PyTorch Dataset implementation that reads YOLO-format
annotations and converts them to Faster R-CNN format for training.
"""

from pathlib import Path

import torch
import torchvision.transforms as t
from PIL import Image
from torch.utils.data import Dataset


class FasterRCNNDataset(Dataset):
    """Dataset for Faster R-CNN training using YOLO-format annotations.

    Reads images and YOLO txt annotations, converting bounding boxes to
    absolute pixel coordinates (x1, y1, x2, y2) as required by Faster R-CNN.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transforms=None,
    ):
        """Initialize Faster R-CNN dataset.

        :param images_dir: Directory containing training images
        :param labels_dir: Directory containing YOLO format labels (.txt)
        :param transforms: Optional torchvision transforms to apply
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms if transforms else t.ToTensor()

        # Collect all image files
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        )

        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        :return: Number of images in dataset
        :rtype: int
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get image and target for training.

        :param idx: Index of sample to retrieve
        :return: Tuple of (image_tensor, target_dict) where target_dict contains:
            - boxes: [num_objects, 4] bounding boxes (x1, y1, x2, y2) in pixels
            - labels: [num_objects] class IDs (1-indexed, 0 is background)
            - image_id: Image identifier
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        """
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Load YOLO format annotations
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    # YOLO format: class_id cx cy w h (normalized)
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx_norm = float(parts[1])
                        cy_norm = float(parts[2])
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])

                        # Convert to absolute pixel coordinates
                        cx = cx_norm * width
                        cy = cy_norm * height
                        w = w_norm * width
                        h = h_norm * height

                        # Convert to (x1, y1, x2, y2) format
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2

                        boxes.append([x1, y1, x2, y2])
                        # Faster R-CNN expects 1-indexed labels (0 is background)
                        labels.append(class_id + 1)

        # Convert to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Empty annotation
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Create target dict
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        # Apply transforms
        image = self.transforms(image)

        return image, target
