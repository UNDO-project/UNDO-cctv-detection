"""Shared pytest fixtures for CCTV detection tests."""

import pytest
from PIL import Image
import numpy as np


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Create temporary dataset directory structure.

    :param tmp_path: pytest's temporary directory fixture
    :return: Temporary dataset directory with images/ and labels/ subdirectories
    :rtype: Path
    """
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    return tmp_path


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing.

    :return: A 640x640 RGB test image
    :rtype: PIL.Image
    """
    # Create a simple test image (640x640 RGB)
    image_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


@pytest.fixture
def sample_image_path(tmp_path, sample_image):
    """Create a sample image file on disk.

    :param tmp_path: pytest's temporary directory fixture
    :param sample_image: Sample PIL Image fixture
    :return: Path to the saved test image
    :rtype: Path
    """
    image_path = tmp_path / "test_image.jpg"
    sample_image.save(image_path)
    return image_path


@pytest.fixture
def sample_label_path(tmp_path):
    """Create a sample YOLO format label file.

    :param tmp_path: pytest's temporary directory fixture
    :return: Path to the saved test label file
    :rtype: Path
    """
    label_path = tmp_path / "test_label.txt"
    # YOLO format: class_id center_x center_y width height
    label_path.write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.15\n")
    return label_path


@pytest.fixture
def mock_dataset_structure(tmp_path):
    """Create a complete mock dataset structure with images and labels.

    :param tmp_path: pytest's temporary directory fixture
    :return: Dictionary with paths to images_dir, labels_dir, and lists of files
    :rtype: dict
    """
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Create 5 sample image/label pairs
    image_paths = []
    label_paths = []

    for i in range(5):
        # Create image
        img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = images_dir / f"image_{i}.jpg"
        img.save(img_path)
        image_paths.append(img_path)

        # Create label
        label_path = labels_dir / f"image_{i}.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.3\n")
        label_paths.append(label_path)

    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "image_paths": image_paths,
        "label_paths": label_paths,
    }
