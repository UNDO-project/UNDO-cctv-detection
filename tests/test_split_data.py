"""Integration tests for split_data module.

This module tests dataset splitting and YAML configuration file creation
for Ultralytics YOLO training.
"""

import pytest
from pathlib import Path
import yaml
from PIL import Image
import numpy as np

from src.split_data import split_dataset, create_data_yaml


class TestSplitDataset:
    """Test split_dataset function."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset with images and labels.

        :param tmp_path: Temporary directory path
        :return: Dictionary with source directories and file count
        """
        source_images = tmp_path / "source_images"
        source_labels = tmp_path / "source_labels"
        source_images.mkdir()
        source_labels.mkdir()

        # Create 10 image-label pairs
        for i in range(10):
            # Create image
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = source_images / f"image_{i:03d}.jpg"
            img.save(img_path)

            # Create label
            label_path = source_labels / f"image_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.3\n")

        return {
            "images_dir": source_images,
            "labels_dir": source_labels,
            "count": 10,
        }

    def test_split_dataset_creates_directory_structure(self, sample_dataset, tmp_path):
        """Test that split_dataset creates proper directory structure.

        :param sample_dataset: Sample dataset fixture
        :param tmp_path: Temporary directory path
        :return: None
        """
        output_dir = tmp_path / "output"

        train_dir, val_dir = split_dataset(
            sample_dataset["images_dir"],
            sample_dataset["labels_dir"],
            train_ratio=0.7,
            output_dir=output_dir,
        )

        # Check that all directories exist
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "images" / "val").exists()
        assert (output_dir / "labels" / "train").exists()
        assert (output_dir / "labels" / "val").exists()

    def test_split_dataset_returns_correct_paths(self, sample_dataset, tmp_path):
        """Test that split_dataset returns correct directory paths.

        :param sample_dataset: Sample dataset fixture
        :param tmp_path: Temporary directory path
        :return: None
        """
        output_dir = tmp_path / "output"

        train_dir, val_dir = split_dataset(
            sample_dataset["images_dir"],
            sample_dataset["labels_dir"],
            train_ratio=0.7,
            output_dir=output_dir,
        )

        assert train_dir == output_dir / "images" / "train"
        assert val_dir == output_dir / "images" / "val"

    def test_split_dataset_respects_train_ratio(self, sample_dataset, tmp_path):
        """Test that split_dataset respects the training ratio.

        :param sample_dataset: Sample dataset fixture
        :param tmp_path: Temporary directory path
        :return: None
        """
        output_dir = tmp_path / "output"
        train_ratio = 0.7

        train_dir, val_dir = split_dataset(
            sample_dataset["images_dir"],
            sample_dataset["labels_dir"],
            train_ratio=train_ratio,
            output_dir=output_dir,
        )

        train_count = len(list(train_dir.glob("*.jpg")))
        val_count = len(list(val_dir.glob("*.jpg")))

        # Total should equal original count
        assert train_count + val_count == sample_dataset["count"]

        # Train ratio should be approximately correct (allow ±1 for rounding)
        expected_train = int(sample_dataset["count"] * train_ratio)
        assert abs(train_count - expected_train) <= 1

    def test_split_dataset_copies_corresponding_labels(self, sample_dataset, tmp_path):
        """Test that labels are copied with their corresponding images.

        :param sample_dataset: Sample dataset fixture
        :param tmp_path: Temporary directory path
        :return: None
        """
        output_dir = tmp_path / "output"

        split_dataset(
            sample_dataset["images_dir"],
            sample_dataset["labels_dir"],
            train_ratio=0.7,
            output_dir=output_dir,
        )

        # Check train split
        train_images = (output_dir / "images" / "train").glob("*.jpg")
        for img_path in train_images:
            label_path = output_dir / "labels" / "train" / f"{img_path.stem}.txt"
            assert label_path.exists(), f"Missing label for {img_path.name}"

        # Check val split
        val_images = (output_dir / "images" / "val").glob("*.jpg")
        for img_path in val_images:
            label_path = output_dir / "labels" / "val" / f"{img_path.stem}.txt"
            assert label_path.exists(), f"Missing label for {img_path.name}"

    def test_split_dataset_handles_different_image_extensions(self, tmp_path):
        """Test splitting dataset with various image extensions.

        :param tmp_path: Temporary directory path
        :return: None
        """
        source_images = tmp_path / "images"
        source_labels = tmp_path / "labels"
        source_images.mkdir()
        source_labels.mkdir()

        # Create images with different extensions
        for ext in [".jpg", ".jpeg", ".png"]:
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = source_images / f"image{ext}"
            img.save(img_path)

            label_path = source_labels / "image.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.3\n")

        output_dir = tmp_path / "output"

        train_dir, val_dir = split_dataset(
            source_images, source_labels, train_ratio=0.7, output_dir=output_dir
        )

        # All images should be processed
        total_images = len(list(train_dir.glob("*"))) + len(list(val_dir.glob("*")))
        assert total_images == 3

    def test_split_dataset_is_deterministic(self, sample_dataset, tmp_path):
        """Test that split_dataset produces consistent results.

        :param sample_dataset: Sample dataset fixture
        :param tmp_path: Temporary directory path
        :return: None
        """
        output_dir1 = tmp_path / "output1"
        output_dir2 = tmp_path / "output2"

        train1, val1 = split_dataset(
            sample_dataset["images_dir"],
            sample_dataset["labels_dir"],
            train_ratio=0.7,
            output_dir=output_dir1,
        )

        train2, val2 = split_dataset(
            sample_dataset["images_dir"],
            sample_dataset["labels_dir"],
            train_ratio=0.7,
            output_dir=output_dir2,
        )

        # Should have same counts due to random_state=42
        assert len(list(train1.glob("*"))) == len(list(train2.glob("*")))
        assert len(list(val1.glob("*"))) == len(list(val2.glob("*")))


class TestCreateDataYaml:
    """Test create_data_yaml function."""

    def test_create_data_yaml_creates_file(self, tmp_path):
        """Test that create_data_yaml creates YAML file.

        :param tmp_path: Temporary directory path
        :return: None
        """
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        yaml_path = tmp_path / "data.yaml"

        train_dir.mkdir()
        val_dir.mkdir()

        create_data_yaml(
            train_dir,
            val_dir,
            nc=2,
            names=["CCTV", "CCTV-SIGNS"],
            output_yaml_path=yaml_path,
        )

        assert yaml_path.exists()

    def test_create_data_yaml_has_correct_structure(self, tmp_path):
        """Test that YAML file has correct structure.

        :param tmp_path: Temporary directory path
        :return: None
        """
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        yaml_path = tmp_path / "data.yaml"

        train_dir.mkdir()
        val_dir.mkdir()

        create_data_yaml(
            train_dir,
            val_dir,
            nc=2,
            names=["CCTV", "CCTV-SIGNS"],
            output_yaml_path=yaml_path,
        )

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        assert "train" in data
        assert "val" in data
        assert "nc" in data
        assert "names" in data

    def test_create_data_yaml_has_correct_values(self, tmp_path):
        """Test that YAML file contains correct values.

        :param tmp_path: Temporary directory path
        :return: None
        """
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        yaml_path = tmp_path / "data.yaml"

        train_dir.mkdir()
        val_dir.mkdir()

        create_data_yaml(
            train_dir,
            val_dir,
            nc=2,
            names=["CCTV", "CCTV-SIGNS"],
            output_yaml_path=yaml_path,
        )

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        assert data["nc"] == 2
        assert data["names"] == ["CCTV", "CCTV-SIGNS"]
        assert str(train_dir.resolve()) in data["train"]
        assert str(val_dir.resolve()) in data["val"]

    def test_create_data_yaml_uses_absolute_paths(self, tmp_path):
        """Test that YAML uses absolute paths.

        :param tmp_path: Temporary directory path
        :return: None
        """
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        yaml_path = tmp_path / "data.yaml"

        train_dir.mkdir()
        val_dir.mkdir()

        create_data_yaml(
            train_dir,
            val_dir,
            nc=2,
            names=["CCTV", "CCTV-SIGNS"],
            output_yaml_path=yaml_path,
        )

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Paths should be absolute
        assert Path(data["train"]).is_absolute()
        assert Path(data["val"]).is_absolute()

    def test_create_data_yaml_with_different_class_counts(self, tmp_path):
        """Test creating YAML with different numbers of classes.

        :param tmp_path: Temporary directory path
        :return: None
        """
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        yaml_path = tmp_path / "data.yaml"

        train_dir.mkdir()
        val_dir.mkdir()

        # Test with 3 classes
        create_data_yaml(
            train_dir,
            val_dir,
            nc=3,
            names=["class1", "class2", "class3"],
            output_yaml_path=yaml_path,
        )

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        assert data["nc"] == 3
        assert len(data["names"]) == 3
