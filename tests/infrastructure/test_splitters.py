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

    def test_split_returns_three_datasets(self, splitter, sample_dataset):
        """Test that split returns train, val, and test datasets."""
        train, val, test = splitter.split(
            sample_dataset, train_ratio=0.7, val_ratio=0.2
        )

        assert train is not None
        assert val is not None
        assert test is not None

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

    def test_split_calculates_test_size_correctly(self, splitter, sample_dataset):
        """Test that test size is calculated as remaining data."""
        train_ratio = 0.6
        val_ratio = 0.2
        train, val, test = splitter.split(
            sample_dataset, train_ratio=train_ratio, val_ratio=val_ratio
        )

        # Test should be remaining: 1 - 0.6 - 0.2 = 0.2
        expected_test_size = int(len(sample_dataset) * (1 - train_ratio - val_ratio))
        assert abs(len(test) - expected_test_size) <= 1

    def test_split_with_different_ratios(self, splitter, sample_dataset):
        """Test split with various ratio combinations."""
        test_cases = [
            (0.7, 0.2),  # 70% train, 20% val, 10% test
            (0.8, 0.1),  # 80% train, 10% val, 10% test
            (0.6, 0.3),  # 60% train, 30% val, 10% test
        ]

        for train_ratio, val_ratio in test_cases:
            train, val, test = splitter.split(
                sample_dataset, train_ratio=train_ratio, val_ratio=val_ratio
            )

            # Total should always equal original dataset size
            assert len(train) + len(val) + len(test) == len(sample_dataset)

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

    def test_split_with_large_dataset(self, splitter):
        """Test split with a larger dataset."""
        large_dataset = list(range(10000))
        train, val, test = splitter.split(large_dataset, train_ratio=0.7, val_ratio=0.2)

        # Verify proportions are correct
        assert len(train) == pytest.approx(7000, abs=10)
        assert len(val) == pytest.approx(2000, abs=10)
        assert len(test) == pytest.approx(1000, abs=10)

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

    def test_split_covers_entire_dataset(self, splitter, sample_dataset):
        """Test that all items from original dataset appear in split."""
        train, val, test = splitter.split(
            sample_dataset, train_ratio=0.7, val_ratio=0.2
        )

        # Union of all splits should equal original dataset
        combined = set(train) | set(val) | set(test)
        original = set(sample_dataset)

        assert combined == original

    def test_split_with_equal_ratios(self, splitter):
        """Test split with equal train/val/test ratios."""
        dataset = list(range(90))  # 90 items for clean division
        train, val, test = splitter.split(dataset, train_ratio=1 / 3, val_ratio=1 / 3)

        # Each split should have approximately 1/3 of data
        assert len(train) == pytest.approx(30, abs=2)
        assert len(val) == pytest.approx(30, abs=2)
        assert len(test) == pytest.approx(30, abs=2)
