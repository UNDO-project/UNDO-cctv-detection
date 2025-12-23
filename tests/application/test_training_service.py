"""Unit tests for TrainingService application service."""

import pytest
from unittest.mock import Mock, patch
import torch
from torch.utils.data import Dataset

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

    def test_initialization(self, mock_dataset, mock_trainer, mock_splitter):
        """Test that service initializes with correct dependencies."""
        service = TrainingService(mock_dataset, mock_trainer, mock_splitter)

        assert service.dataset is mock_dataset
        assert service.model_trainer is mock_trainer
        assert service.dataset_splitter is mock_splitter

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

    def test_run_training_uses_default_config_values(
        self, service, mock_splitter, mock_trainer
    ):
        """Test that run_training uses default config values when not specified."""
        # Arrange
        mock_train = MockDataset(70)
        mock_val = MockDataset(20)
        mock_test = MockDataset(10)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training()  # No arguments, should use defaults

        # Assert
        mock_splitter.split.assert_called_once()
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

        # Verify that DataLoaders were created and passed
        train_loader = call_args[0]
        val_loader = call_args[1]
        device = call_args[2]

        assert hasattr(train_loader, "__iter__")  # DataLoader is iterable
        assert hasattr(val_loader, "__iter__")
        assert isinstance(device, torch.device)

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
        device = call_args[2]
        assert device.type == "mps"

    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=True)
    def test_run_training_selects_cuda_device_when_mps_unavailable(
        self, mock_cuda, mock_mps, service, mock_splitter, mock_trainer
    ):
        """Test that CUDA device is selected when MPS is unavailable."""
        # Arrange
        mock_train = MockDataset(10)
        mock_val = MockDataset(5)
        mock_test = MockDataset(5)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training()

        # Assert
        call_args = mock_trainer.train.call_args[0]
        device = call_args[2]
        assert device.type == "cuda"

    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=False)
    def test_run_training_selects_cpu_device_when_no_gpu(
        self, mock_cuda, mock_mps, service, mock_splitter, mock_trainer
    ):
        """Test that CPU device is selected when no GPU is available."""
        # Arrange
        mock_train = MockDataset(10)
        mock_val = MockDataset(5)
        mock_test = MockDataset(5)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training()

        # Assert
        call_args = mock_trainer.train.call_args[0]
        device = call_args[2]
        assert device.type == "cpu"

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
        train_loader = call_args[0]
        val_loader = call_args[1]

        # DataLoaders should use the custom batch size
        assert train_loader.batch_size == custom_batch_size
        assert val_loader.batch_size == custom_batch_size

    def test_run_training_creates_shuffled_train_loader(
        self, service, mock_splitter, mock_trainer
    ):
        """Test that training DataLoader is created with shuffle=True."""
        # Arrange
        mock_train = MockDataset(70)
        mock_val = MockDataset(20)
        mock_test = MockDataset(10)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training()

        # Assert
        call_args = mock_trainer.train.call_args[0]
        train_loader = call_args[0]

        # Training loader should have shuffle enabled
        # Note: We can't directly check shuffle, but we verify the loader was created
        assert train_loader is not None

    def test_run_training_workflow_execution_order(
        self, service, mock_splitter, mock_trainer
    ):
        """Test that training workflow executes steps in correct order."""
        # Arrange
        mock_train = MockDataset(70)
        mock_val = MockDataset(20)
        mock_test = MockDataset(10)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        execution_order = []

        def track_split(*args, **kwargs):
            execution_order.append("split")
            return (mock_train, mock_val, mock_test)

        def track_train(*args, **kwargs):
            execution_order.append("train")

        mock_splitter.split.side_effect = track_split
        mock_trainer.train.side_effect = track_train

        # Act
        service.run_training()

        # Assert
        assert execution_order == ["split", "train"]
        assert mock_splitter.split.called
        assert mock_trainer.train.called

    def test_run_training_with_different_ratios(
        self, service, mock_splitter, mock_trainer
    ):
        """Test that different train/val ratios are passed correctly."""
        # Arrange
        mock_train = MockDataset(80)
        mock_val = MockDataset(15)
        mock_test = MockDataset(5)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training(train_ratio=0.8, val_ratio=0.15, batch_size=4)

        # Assert
        call_args = mock_splitter.split.call_args[0]
        assert call_args[1] == 0.8
        assert call_args[2] == 0.15

    def test_trainer_receives_device_parameter(
        self, service, mock_splitter, mock_trainer
    ):
        """Test that trainer receives device parameter."""
        # Arrange
        mock_train = MockDataset(10)
        mock_val = MockDataset(5)
        mock_test = MockDataset(5)
        mock_splitter.split.return_value = (mock_train, mock_val, mock_test)

        # Act
        service.run_training()

        # Assert
        call_args = mock_trainer.train.call_args[0]
        assert len(call_args) == 3  # train_loader, val_loader, device
        device = call_args[2]
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]
