"""Unit tests for TrainingService application service."""

from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import Dataset, RandomSampler

from src.application.training_service import TrainingService


class MockDataset(Dataset):
    """Mock dataset for testing."""

    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"image": torch.randn(3, 64, 64), "label": 0}


class TestTrainingService:
    """Test TrainingService application service."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        return MockDataset(size=100)

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock ModelTrainer."""
        return Mock()

    @pytest.fixture
    def mock_splitter(self):
        """Create a mock DatasetSplitter."""
        return Mock()

    @pytest.fixture
    def service(self, mock_dataset, mock_trainer, mock_splitter):
        """Create TrainingService with mocked dependencies."""
        return TrainingService(mock_dataset, mock_trainer, mock_splitter)

    def test_run_training_splits_dataset(self, service, mock_splitter):
        """Test that run_training calls dataset splitter with correct ratios."""
        # Arrange
        mock_train = MockDataset(70)
        mock_val = MockDataset(20)
        mock_test = MockDataset(10)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training(train_ratio=0.7, val_ratio=0.2, batch_size=4)

        # Assert
        mock_splitter.split.assert_called_once()
        call_args = mock_splitter.split.call_args
        assert call_args[0][0] is service.dataset
        assert call_args[0][1] == 0.7  # train_ratio
        assert call_args[0][2] == 0.2  # val_ratio

        # Should use TRAIN_RATIO (0.7) and VAL_RATIO (0.3) from config

    def test_run_training_calls_trainer_with_dataloaders(
        self, service, mock_splitter, mock_trainer
    ):
        """Test that run_training calls trainer with DataLoaders."""
        # Arrange
        mock_train = MockDataset(70)
        mock_val = MockDataset(20)
        mock_test = MockDataset(10)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training(batch_size=4)

        # Assert
        mock_trainer.train.assert_called_once()
        call_args = mock_trainer.train.call_args[0]

        # Verify that device is first, then DataLoaders (new interface)
        device = call_args[0]
        train_loader = call_args[1]
        val_loader = call_args[2]

        assert isinstance(device, torch.device)
        assert hasattr(train_loader, "__iter__")  # DataLoader is iterable
        assert hasattr(val_loader, "__iter__")

    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_run_training_selects_mps_device_when_available(
        self, mock_cuda, mock_mps, service, mock_splitter, mock_trainer
    ):
        """Test that MPS device is selected when available."""
        # Arrange
        mock_train = MockDataset(10)
        mock_val = MockDataset(5)
        mock_test = MockDataset(5)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training()

        # Assert
        call_args = mock_trainer.train.call_args[0]
        device = call_args[0]  # Device is now first parameter
        assert device.type == "mps"

    def test_run_training_with_custom_batch_size(
        self, service, mock_splitter, mock_trainer
    ):
        """Test that run_training respects custom batch size."""
        # Arrange
        mock_train = MockDataset(80)
        mock_val = MockDataset(20)
        mock_test = MockDataset(10)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        custom_batch_size = 16
        service.run_training(batch_size=custom_batch_size)

        # Assert
        call_args = mock_trainer.train.call_args[0]
        # New interface: device is first, then loaders
        train_loader = call_args[1]
        val_loader = call_args[2]

        # DataLoaders should use the custom batch size
        assert train_loader.batch_size == custom_batch_size
        assert val_loader.batch_size == custom_batch_size

    def test_run_training_creates_shuffled_train_loader(
        self, service, mock_splitter, mock_trainer
    ):
        """Test that the train loader is built with a shuffling sampler."""
        # Arrange
        mock_train = MockDataset(70)
        mock_val = MockDataset(20)
        mock_test = MockDataset(10)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training()

        # Assert - shuffle=True gives a RandomSampler
        _, train_loader, val_loader = mock_trainer.train.call_args[0]
        assert isinstance(train_loader.sampler, RandomSampler)
        assert train_loader.dataset is mock_train
        assert val_loader.dataset is mock_val
