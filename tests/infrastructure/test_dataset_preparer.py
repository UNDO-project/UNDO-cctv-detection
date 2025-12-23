"""Integration tests for SklearnDatasetPreparer."""

import pytest
from PIL import Image
import numpy as np

from src.infrastructure.dataset_preparer_impl import SklearnDatasetPreparer


class TestSklearnDatasetPreparer:
    """Test SklearnDatasetPreparer infrastructure implementation."""

    @pytest.fixture
    def preparer(self):
        """Create SklearnDatasetPreparer instance."""
        return SklearnDatasetPreparer()

    @pytest.fixture
    def dataset_with_labels(self, tmp_path):
        """Create a mock dataset with images and labels."""
        source_images = tmp_path / "source_images"
        source_labels = tmp_path / "source_labels"
        source_images.mkdir()
        source_labels.mkdir()

        # Create 10 image-label pairs
        for i in range(10):
            # Create image
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = source_images / f"image_{i}.jpg"
            img.save(img_path)

            # Create corresponding label
            label_path = source_labels / f"image_{i}.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.3\n")

        return {
            "source_images": source_images,
            "source_labels": source_labels,
            "image_count": 10,
        }

    def test_prepare_ultralytics_dataset_creates_directories(
        self, preparer, dataset_with_labels, tmp_path
    ):
        """Test that prepare_ultralytics_dataset creates output directories."""
        output_images = tmp_path / "output_images"
        output_labels = tmp_path / "output_labels"

        preparer.prepare_ultralytics_dataset(
            source_images=dataset_with_labels["source_images"],
            source_labels=dataset_with_labels["source_labels"],
            output_images=output_images,
            output_labels=output_labels,
            train_ratio=0.7,
            val_ratio=0.2,
            move_files=False,
        )

        # Check that split directories were created
        assert (output_images / "train").exists()
        assert (output_images / "val").exists()
        assert (output_images / "test").exists()
        assert (output_labels / "train").exists()
        assert (output_labels / "val").exists()
        assert (output_labels / "test").exists()

    def test_prepare_ultralytics_dataset_splits_files(
        self, preparer, dataset_with_labels, tmp_path
    ):
        """Test that files are split into train/val/test directories."""
        output_images = tmp_path / "output_images"
        output_labels = tmp_path / "output_labels"

        preparer.prepare_ultralytics_dataset(
            source_images=dataset_with_labels["source_images"],
            source_labels=dataset_with_labels["source_labels"],
            output_images=output_images,
            output_labels=output_labels,
            train_ratio=0.7,
            val_ratio=0.2,
            move_files=False,
        )

        # Count files in each split
        train_images = list((output_images / "train").glob("*.jpg"))
        val_images = list((output_images / "val").glob("*.jpg"))
        test_images = list((output_images / "test").glob("*.jpg"))

        # Total should equal original count
        total_images = len(train_images) + len(val_images) + len(test_images)
        assert total_images == dataset_with_labels["image_count"]

    def test_prepare_ultralytics_dataset_preserves_image_label_pairs(
        self, preparer, dataset_with_labels, tmp_path
    ):
        """Test that each image has a corresponding label."""
        output_images = tmp_path / "output_images"
        output_labels = tmp_path / "output_labels"

        preparer.prepare_ultralytics_dataset(
            source_images=dataset_with_labels["source_images"],
            source_labels=dataset_with_labels["source_labels"],
            output_images=output_images,
            output_labels=output_labels,
            train_ratio=0.7,
            val_ratio=0.2,
            move_files=False,
        )

        # Check each split
        for split in ["train", "val", "test"]:
            images = list((output_images / split).glob("*.jpg"))
            for img in images:
                label = output_labels / split / f"{img.stem}.txt"
                assert label.exists(), f"Missing label for {img.name} in {split}"

    def test_prepare_ultralytics_dataset_respects_train_ratio(
        self, preparer, dataset_with_labels, tmp_path
    ):
        """Test that training ratio is approximately respected."""
        output_images = tmp_path / "output_images"
        output_labels = tmp_path / "output_labels"

        train_ratio = 0.7
        preparer.prepare_ultralytics_dataset(
            source_images=dataset_with_labels["source_images"],
            source_labels=dataset_with_labels["source_labels"],
            output_images=output_images,
            output_labels=output_labels,
            train_ratio=train_ratio,
            val_ratio=0.2,
            move_files=False,
        )

        train_images = list((output_images / "train").glob("*.jpg"))
        expected_count = int(dataset_with_labels["image_count"] * train_ratio)

        # Allow for rounding differences
        assert abs(len(train_images) - expected_count) <= 1

    def test_prepare_ultralytics_dataset_with_copy_mode(
        self, preparer, dataset_with_labels, tmp_path
    ):
        """Test that copy mode preserves original files."""
        output_images = tmp_path / "output_images"
        output_labels = tmp_path / "output_labels"

        original_image_count = len(
            list(dataset_with_labels["source_images"].glob("*.*"))
        )

        preparer.prepare_ultralytics_dataset(
            source_images=dataset_with_labels["source_images"],
            source_labels=dataset_with_labels["source_labels"],
            output_images=output_images,
            output_labels=output_labels,
            train_ratio=0.7,
            val_ratio=0.2,
            move_files=False,  # Copy mode
        )

        # Original files should still exist
        remaining_images = len(list(dataset_with_labels["source_images"].glob("*.*")))
        assert remaining_images == original_image_count

    def test_prepare_ultralytics_dataset_with_move_mode(
        self, preparer, dataset_with_labels, tmp_path
    ):
        """Test that move mode removes original files."""
        output_images = tmp_path / "output_images"
        output_labels = tmp_path / "output_labels"

        preparer.prepare_ultralytics_dataset(
            source_images=dataset_with_labels["source_images"],
            source_labels=dataset_with_labels["source_labels"],
            output_images=output_images,
            output_labels=output_labels,
            train_ratio=0.7,
            val_ratio=0.2,
            move_files=True,  # Move mode
        )

        # Original directories should be empty
        remaining_images = list(dataset_with_labels["source_images"].glob("*.*"))
        remaining_labels = list(dataset_with_labels["source_labels"].glob("*.*"))

        assert len(remaining_images) == 0
        assert len(remaining_labels) == 0

    def test_prepare_ultralytics_dataset_handles_missing_labels(
        self, preparer, tmp_path
    ):
        """Test that images without labels are skipped."""
        source_images = tmp_path / "source_images"
        source_labels = tmp_path / "source_labels"
        source_images.mkdir()
        source_labels.mkdir()

        # Create 10 images but only 6 labels
        for i in range(10):
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = source_images / f"image_{i}.jpg"
            img.save(img_path)

        # Only create labels for images 0-5 (6 labels total)
        for i in range(6):
            label_path = source_labels / f"image_{i}.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.3\n")

        output_images = tmp_path / "output_images"
        output_labels = tmp_path / "output_labels"

        preparer.prepare_ultralytics_dataset(
            source_images=source_images,
            source_labels=source_labels,
            output_images=output_images,
            output_labels=output_labels,
            train_ratio=0.7,
            val_ratio=0.2,
            move_files=False,
        )

        # Count total processed files (should be 6, not 10)
        train_images = list((output_images / "train").glob("*.jpg"))
        val_images = list((output_images / "val").glob("*.jpg"))
        test_images = list((output_images / "test").glob("*.jpg"))

        total = len(train_images) + len(val_images) + len(test_images)
        assert total == 6  # Only images with labels

    def test_prepare_ultralytics_dataset_with_empty_source(self, preparer, tmp_path):
        """Test behavior with empty source directory."""
        source_images = tmp_path / "empty_images"
        source_labels = tmp_path / "empty_labels"
        source_images.mkdir()
        source_labels.mkdir()

        output_images = tmp_path / "output_images"
        output_labels = tmp_path / "output_labels"

        # Should handle gracefully
        preparer.prepare_ultralytics_dataset(
            source_images=source_images,
            source_labels=source_labels,
            output_images=output_images,
            output_labels=output_labels,
            train_ratio=0.7,
            val_ratio=0.2,
            move_files=False,
        )

        # No output directories should be created if no data
        # (or they exist but are empty - both behaviors are acceptable)

    def test_prepare_ultralytics_dataset_with_different_ratios(
        self, preparer, dataset_with_labels, tmp_path
    ):
        """Test dataset preparation with different split ratios."""
        output_images = tmp_path / "output_images"
        output_labels = tmp_path / "output_labels"

        preparer.prepare_ultralytics_dataset(
            source_images=dataset_with_labels["source_images"],
            source_labels=dataset_with_labels["source_labels"],
            output_images=output_images,
            output_labels=output_labels,
            train_ratio=0.8,
            val_ratio=0.1,
            move_files=False,
        )

        train_count = len(list((output_images / "train").glob("*.jpg")))
        val_count = len(list((output_images / "val").glob("*.jpg")))
        test_count = len(list((output_images / "test").glob("*.jpg")))

        # Check total is preserved
        assert (
            train_count + val_count + test_count == dataset_with_labels["image_count"]
        )

        # Check approximate ratios
        assert train_count >= 7  # ~80% of 10
        assert val_count <= 2  # ~10% of 10
