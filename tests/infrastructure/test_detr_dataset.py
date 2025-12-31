"""Unit tests for DETR dataset and data preparation utilities.

Tests the DETRDataset and DETRDataPreparer classes to ensure correct conversion
from YOLO format to COCO/DETR format for transformer-based object detection.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.infrastructure.detr_data_preparer import DETRDataPreparer
from src.infrastructure.detr_dataset import DETRDataset


class TestDETRDataPreparer:
    """Test cases for DETRDataPreparer class."""

    def test_initialization(self):
        """Test that data preparer initializes with class names."""
        class_names = ["CCTV", "CCTV-SIGNS"]
        preparer = DETRDataPreparer(class_names=class_names)

        assert preparer.class_names == class_names
        assert preparer.class_to_idx == {"CCTV": 0, "CCTV-SIGNS": 1}

    def test_yolo_to_coco_bbox_conversion(self):
        """Test YOLO to COCO bounding box format conversion."""
        preparer = DETRDataPreparer(class_names=["CCTV"])

        # YOLO format: [cx, cy, w, h] normalized
        yolo_bbox = [0.5, 0.5, 0.2, 0.3]
        img_width = 640
        img_height = 480

        # Expected COCO format: [x, y, w, h] in absolute pixels
        coco_bbox = preparer.yolo_to_coco_bbox(yolo_bbox, img_width, img_height)

        # Center: (0.5 * 640, 0.5 * 480) = (320, 240)
        # Size: (0.2 * 640, 0.3 * 480) = (128, 144)
        # Top-left: (320 - 128/2, 240 - 144/2) = (256, 168)
        expected = [256.0, 168.0, 128.0, 144.0]

        assert coco_bbox == pytest.approx(expected)

    def test_yolo_to_coco_bbox_at_origin(self):
        """Test conversion for bbox at image origin."""
        preparer = DETRDataPreparer(class_names=["CCTV"])

        # Bbox at top-left corner
        yolo_bbox = [0.1, 0.1, 0.2, 0.2]
        img_width = 1000
        img_height = 1000

        coco_bbox = preparer.yolo_to_coco_bbox(yolo_bbox, img_width, img_height)

        # Center: (100, 100), Size: (200, 200), Top-left: (0, 0)
        expected = [0.0, 0.0, 200.0, 200.0]

        assert coco_bbox == pytest.approx(expected)

    def test_prepare_annotations_with_no_labels(self, tmp_path: Path):
        """Test that missing label files are handled gracefully."""
        preparer = DETRDataPreparer(class_names=["CCTV"])

        # Create test image without corresponding label
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create dummy image
        img = Image.new("RGB", (640, 480), color="red")
        img_path = images_dir / "test.jpg"
        img.save(img_path)

        # Prepare annotations (should skip image with no label)
        annotations = preparer.prepare_annotations(images_dir, labels_dir)

        assert len(annotations) == 0

    def test_prepare_annotations_with_valid_labels(self, tmp_path: Path):
        """Test annotation preparation with valid YOLO labels."""
        preparer = DETRDataPreparer(class_names=["CCTV", "CCTV-SIGNS"])

        # Create test directories
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create dummy image
        img = Image.new("RGB", (640, 480), color="red")
        img_path = images_dir / "test.jpg"
        img.save(img_path)

        # Create YOLO label file
        label_path = labels_dir / "test.txt"
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.3\n")  # class_id cx cy w h
            f.write("1 0.3 0.3 0.1 0.1\n")

        # Prepare annotations
        annotations = preparer.prepare_annotations(images_dir, labels_dir)

        assert len(annotations) == 1
        ann = annotations[0]

        assert ann["image_id"] == 0
        assert str(img_path) == ann["image_path"]
        assert len(ann["boxes"]) == 2
        assert len(ann["category_ids"]) == 2
        assert ann["category_ids"] == [0, 1]
        assert len(ann["area"]) == 2
        assert len(ann["iscrowd"]) == 2

    def test_prepare_annotations_with_invalid_lines(self, tmp_path: Path):
        """Test that invalid label lines are skipped."""
        preparer = DETRDataPreparer(class_names=["CCTV"])

        # Create test directories
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create dummy image
        img = Image.new("RGB", (640, 480), color="red")
        img_path = images_dir / "test.jpg"
        img.save(img_path)

        # Create label file with invalid lines
        label_path = labels_dir / "test.txt"
        with open(label_path, "w") as f:
            f.write("invalid line\n")
            f.write("0 0.5\n")  # Too few values
            f.write("\n")  # Empty line

        # Prepare annotations (should skip due to no valid boxes)
        annotations = preparer.prepare_annotations(images_dir, labels_dir)

        assert len(annotations) == 0


class TestDETRDataset:
    """Test cases for DETRDataset class."""

    def test_initialization(self):
        """Test that dataset initializes with required parameters."""
        mock_processor = MagicMock()
        image_paths = ["img1.jpg", "img2.jpg"]
        annotations = [
            {"image_id": 0, "boxes": [[0, 0, 10, 10]], "category_ids": [0]},
            {"image_id": 1, "boxes": [[5, 5, 15, 15]], "category_ids": [1]},
        ]

        dataset = DETRDataset(
            image_paths=image_paths,
            annotations=annotations,
            processor=mock_processor,
            max_size=800,
        )

        assert len(dataset) == 2
        assert dataset.image_paths == image_paths
        assert dataset.annotations == annotations
        assert dataset.processor == mock_processor
        assert dataset.max_size == 800

    def test_len(self):
        """Test dataset length method."""
        mock_processor = MagicMock()
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        annotations = [{} for _ in range(3)]

        dataset = DETRDataset(
            image_paths=image_paths,
            annotations=annotations,
            processor=mock_processor,
        )

        assert len(dataset) == 3

    @patch("src.infrastructure.detr_dataset.Image.open")
    def test_getitem(self, mock_image_open):
        """Test getting an item from the dataset."""
        # Setup mock image
        mock_img = MagicMock(spec=Image.Image)
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img

        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "pixel_values": torch.randn(1, 3, 800, 800),
            "labels": [
                {
                    "class_labels": torch.tensor([0]),
                    "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
                }
            ],
        }

        # Create dataset
        image_paths = ["test.jpg"]
        annotations = [
            {
                "image_id": 0,
                "boxes": [[100.0, 100.0, 200.0, 200.0]],
                "category_ids": [0],
                "area": [10000.0],
                "iscrowd": [0],
            }
        ]

        dataset = DETRDataset(
            image_paths=image_paths,
            annotations=annotations,
            processor=mock_processor,
        )

        # Get item
        item = dataset[0]

        # Verify processor was called
        mock_processor.assert_called_once()
        assert item["pixel_values"].shape == (3, 800, 800)
        assert "class_labels" in item["labels"]

    def test_empty_dataset(self):
        """Test dataset with no samples."""
        mock_processor = MagicMock()
        dataset = DETRDataset(
            image_paths=[],
            annotations=[],
            processor=mock_processor,
        )

        assert len(dataset) == 0
