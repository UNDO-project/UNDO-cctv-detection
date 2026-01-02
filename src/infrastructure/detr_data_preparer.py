"""Prepare CCTV dataset for DETR training.

This module provides utilities to convert YOLO-format annotations to COCO-format
annotations suitable for DETR transformer-based object detection.
"""

from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image


class DETRDataPreparer:
    """Converts YOLO-format dataset to DETR-compatible format.

    YOLO format uses normalized coordinates (cx, cy, w, h) where values are
    in [0, 1] range. COCO format uses absolute pixel coordinates (x, y, w, h)
    where (x, y) is the top-left corner.
    """

    def __init__(self, class_names: list[str]):
        """Initialize with class names.

        :param class_names: List of class names (e.g., ["CCTV", "CCTV-SIGNS"])
        """
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    @staticmethod
    def yolo_to_coco_bbox(
        yolo_bbox: list[float],
        img_width: int,
        img_height: int,
    ) -> list[float]:
        """Convert YOLO bbox (cx, cy, w, h) to COCO format (x, y, w, h).

        YOLO format is normalized [0, 1], COCO is absolute pixels.

        :param yolo_bbox: [center_x, center_y, width, height] normalized
        :param img_width: Image width in pixels
        :param img_height: Image height in pixels
        :return: [x, y, width, height] in pixels (COCO format)
        :rtype: List[float]
        """
        cx, cy, w, h = yolo_bbox
        # Convert to absolute coordinates
        cx *= img_width
        cy *= img_height
        w *= img_width
        h *= img_height
        # Convert to top-left corner
        x = cx - w / 2
        y = cy - h / 2
        return [x, y, w, h]

    def prepare_annotations(
        self,
        images_dir: Path,
        labels_dir: Path,
    ) -> list[dict[str, Any]]:
        """Prepare COCO-format annotations from YOLO labels.

        :param images_dir: Directory containing images
        :param labels_dir: Directory containing YOLO .txt labels
        :return: List of annotation dicts in COCO format
        :rtype: List[Dict[str, Any]]
        """
        annotations = []
        image_id = 0

        for img_path in sorted(images_dir.glob("*.jpg")):
            label_path = labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                logger.warning(f"No label found for {img_path.name}")
                continue

            # Load image to get dimensions
            img = Image.open(img_path)
            img_width, img_height = img.size

            # Parse YOLO labels
            boxes = []
            category_ids = []

            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    yolo_bbox = [float(x) for x in parts[1:5]]

                    # Convert to COCO format
                    coco_bbox = self.yolo_to_coco_bbox(yolo_bbox, img_width, img_height)

                    boxes.append(coco_bbox)
                    category_ids.append(class_id)

            # Include images with no boxes as negative samples
            # This helps reduce false positives during training
            if not boxes:
                logger.debug(f"Including negative sample (no boxes): {label_path.name}")

            # Create annotation (including empty boxes for negative samples)
            annotations.append(
                {
                    "image_id": image_id,
                    "image_path": str(img_path),
                    "boxes": boxes,  # Empty list for negative samples
                    "category_ids": category_ids,  # Empty list for negative samples
                    "area": [b[2] * b[3] for b in boxes],  # width * height
                    "iscrowd": [0] * len(boxes),  # No crowd annotations
                }
            )

            image_id += 1

        logger.info(f"Prepared {len(annotations)} DETR annotations")
        return annotations
