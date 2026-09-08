"""Integration tests for SklearnDatasetSplitter."""

import pytest

from src.infrastructure.splitters import SklearnDatasetSplitter


class TestSklearnDatasetSplitter:
    """Test SklearnDatasetSplitter infrastructure implementation."""

    @pytest.fixture
    def splitter(self):
        """Create SklearnDatasetSplitter instance."""
        return SklearnDatasetSplitter()

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset (list of items)."""
        return list(range(100))  # Dataset with 100 items

    def test_split_preserves_total_count(self, splitter, sample_dataset):
        """Test that split preserves total number of items."""
        train, val, test = splitter.split(
            sample_dataset, train_ratio=0.7, val_ratio=0.2
        )

        total_after_split = len(train) + len(val) + len(test)
        assert total_after_split == len(sample_dataset)

    def test_split_respects_train_ratio(self, splitter, sample_dataset):
        """Test that split respects the training ratio."""
        train_ratio = 0.7
        train, val, test = splitter.split(
            sample_dataset, train_ratio=train_ratio, val_ratio=0.2
        )

        expected_train_size = int(len(sample_dataset) * train_ratio)
        # Allow for rounding differences
        assert abs(len(train) - expected_train_size) <= 1

    def test_split_respects_val_ratio(self, splitter, sample_dataset):
        """Test that split respects the validation ratio."""
        val_ratio = 0.2
        train, val, test = splitter.split(
            sample_dataset, train_ratio=0.7, val_ratio=val_ratio
        )

        expected_val_size = int(len(sample_dataset) * val_ratio)
        # Allow for rounding differences
        assert abs(len(val) - expected_val_size) <= 1

    def test_split_is_deterministic(self, splitter, sample_dataset):
        """Test that split produces consistent results (due to random_state=42)."""
        train1, val1, test1 = splitter.split(
            sample_dataset, train_ratio=0.7, val_ratio=0.2
        )
        train2, val2, test2 = splitter.split(
            sample_dataset, train_ratio=0.7, val_ratio=0.2
        )

        # Results should be identical due to fixed random_state
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_split_with_small_dataset(self, splitter):
        """Test split with a small dataset."""
        small_dataset = list(range(10))
        train, val, test = splitter.split(small_dataset, train_ratio=0.7, val_ratio=0.2)

        # Should still produce all three splits
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(small_dataset)

    def test_split_creates_non_overlapping_sets(self, splitter, sample_dataset):
        """Test that train, val, and test sets don't overlap."""
        train, val, test = splitter.split(
            sample_dataset, train_ratio=0.7, val_ratio=0.2
        )

        # Convert to sets to check for overlaps
        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        # No overlaps
        assert len(train_set & val_set) == 0, "Train and val sets overlap"
        assert len(train_set & test_set) == 0, "Train and test sets overlap"
        assert len(val_set & test_set) == 0, "Val and test sets overlap"
