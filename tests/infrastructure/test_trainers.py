"""Unit tests for ModelTrainer implementations."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.infrastructure.trainers import FasterRCNNTrainer, YoloUltralyticsTrainer


class TestModelTrainerInterface:
    """Test ModelTrainer interface contract."""

    def test_validate_inputs_raises_when_no_inputs_provided(self):
        """Test that validate_inputs raises ValueError when neither loaders nor config provided."""
        # Arrange
        trainer = FasterRCNNTrainer(num_classes=2, epochs=1)

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Either \\(train_loader AND val_loader\\) or data_config must be provided",
        ):
            trainer.validate_inputs(None, None, None)

    def test_validate_inputs_accepts_loaders(self):
        """Test that validate_inputs accepts train and val loaders."""
        # Arrange
        trainer = FasterRCNNTrainer(num_classes=2, epochs=1)
        dataset = TensorDataset(torch.randn(10, 3, 64, 64))
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        # Act & Assert (should not raise)
        trainer.validate_inputs(train_loader, val_loader, None)

    def test_validate_inputs_accepts_data_config(self):
        """Test that validate_inputs accepts data config path."""
        # Arrange
        trainer = FasterRCNNTrainer(num_classes=2, epochs=1)
        data_config = Path("/path/to/data.yaml")

        # Act & Assert (should not raise)
        trainer.validate_inputs(None, None, data_config)

    def test_validate_inputs_rejects_only_train_loader(self):
        """Test that validate_inputs rejects train_loader without val_loader."""
        # Arrange
        trainer = FasterRCNNTrainer(num_classes=2, epochs=1)
        dataset = TensorDataset(torch.randn(10, 3, 64, 64))
        train_loader = DataLoader(dataset, batch_size=2)

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Either \\(train_loader AND val_loader\\) or data_config must be provided",
        ):
            trainer.validate_inputs(train_loader, None, None)

    def test_validate_inputs_rejects_only_val_loader(self):
        """Test that validate_inputs rejects val_loader without train_loader."""
        # Arrange
        trainer = FasterRCNNTrainer(num_classes=2, epochs=1)
        dataset = TensorDataset(torch.randn(10, 3, 64, 64))
        val_loader = DataLoader(dataset, batch_size=2)

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Either \\(train_loader AND val_loader\\) or data_config must be provided",
        ):
            trainer.validate_inputs(None, val_loader, None)


class TestYoloUltralyticsTrainer:
    """Test YoloUltralyticsTrainer implementation."""

    @pytest.fixture
    def data_config(self, tmp_path):
        """Create a temporary data config file."""
        config_file = tmp_path / "data.yaml"
        config_file.write_text("train: /path/to/train\nval: /path/to/val\nnc: 2\n")
        return config_file

    @pytest.fixture
    def trainer(self, data_config):
        """Create a YoloUltralyticsTrainer instance."""
        with patch("src.infrastructure.trainers.YOLO"):
            return YoloUltralyticsTrainer(
                model_weights="yolov8n.pt",
                data_config=data_config,
                epochs=1,
                img_size=640,
            )

    def test_train_raises_when_no_data_config(self):
        """Test that train raises ValueError when no data_config provided."""
        # Arrange - create trainer WITHOUT data_config
        with patch("src.infrastructure.trainers.YOLO"):
            trainer = YoloUltralyticsTrainer(
                model_weights="yolov8n.pt",
                data_config=None,  # No data_config in initialization
                epochs=1,
                img_size=640,
            )
        device = torch.device("cpu")

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Either \\(train_loader AND val_loader\\) or data_config must be provided",
        ):
            trainer.train(device, train_loader=None, val_loader=None, data_config=None)

    def test_train_uses_provided_data_config(self, trainer, tmp_path):
        """Test that train uses data_config when provided."""
        # Arrange
        device = torch.device("cpu")
        custom_config = tmp_path / "custom.yaml"
        custom_config.write_text("train: /custom/train\nval: /custom/val\nnc: 2\n")

        with patch.object(trainer.model, "train") as mock_train:
            # Act
            trainer.train(device, data_config=custom_config)

            # Assert
            mock_train.assert_called_once()
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs["data"] == custom_config

    def test_train_uses_instance_data_config_by_default(self, trainer, data_config):
        """Test that train uses instance data_config when none provided."""
        # Arrange
        device = torch.device("cpu")

        with patch.object(trainer.model, "train") as mock_train:
            # Act
            trainer.train(device)

            # Assert
            mock_train.assert_called_once()
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs["data"] == data_config

    def test_train_ignores_dataloaders(self, trainer, data_config):
        """Test that train works even when loaders are provided (they're ignored)."""
        # Arrange
        device = torch.device("cpu")
        dataset = TensorDataset(torch.randn(10, 3, 64, 64))
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        with patch.object(trainer.model, "train") as mock_train:
            # Act
            trainer.train(device, train_loader, val_loader, data_config)

            # Assert - should succeed and use data_config
            mock_train.assert_called_once()
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs["data"] == data_config


class TestFasterRCNNTrainer:
    """Test FasterRCNNTrainer implementation."""

    @pytest.fixture
    def trainer(self):
        """Create a FasterRCNNTrainer instance."""
        return FasterRCNNTrainer(num_classes=3, epochs=1, learning_rate=0.001)

    @pytest.fixture
    def mock_dataloaders(self):
        """Create mock dataloaders with proper structure."""
        # Create mock images and targets
        images = [torch.randn(3, 64, 64) for _ in range(4)]
        targets = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
            for _ in range(4)
        ]

        train_loader = Mock()
        train_loader.__iter__ = Mock(return_value=iter([(images, targets)]))
        train_loader.__len__ = Mock(return_value=1)

        val_loader = Mock()
        val_loader.__iter__ = Mock(return_value=iter([(images, targets)]))
        val_loader.__len__ = Mock(return_value=1)

        return train_loader, val_loader

    def test_train_raises_when_no_loaders(self, trainer):
        """Test that train raises ValueError when loaders not provided."""
        # Arrange
        device = torch.device("cpu")

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Either \\(train_loader AND val_loader\\) or data_config must be provided",
        ):
            trainer.train(device, train_loader=None, val_loader=None)

    def test_train_raises_when_only_train_loader(self, trainer, mock_dataloaders):
        """Test that train raises ValueError when only train_loader provided."""
        # Arrange
        device = torch.device("cpu")
        train_loader, _ = mock_dataloaders

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Either \\(train_loader AND val_loader\\) or data_config must be provided",
        ):
            trainer.train(device, train_loader=train_loader, val_loader=None)

    def test_train_raises_when_only_val_loader(self, trainer, mock_dataloaders):
        """Test that train raises ValueError when only val_loader provided."""
        # Arrange
        device = torch.device("cpu")
        _, val_loader = mock_dataloaders

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Either \\(train_loader AND val_loader\\) or data_config must be provided",
        ):
            trainer.train(device, train_loader=None, val_loader=val_loader)

    def test_train_accepts_both_loaders(self, trainer, mock_dataloaders):
        """Test that train works when both loaders provided."""
        # Arrange
        device = torch.device("cpu")
        train_loader, val_loader = mock_dataloaders

        # Mock the model's forward pass to return losses with gradients
        loss_classifier = torch.tensor(0.5, requires_grad=True)
        loss_box_reg = torch.tensor(0.3, requires_grad=True)

        with (
            patch.object(
                trainer.model,
                "forward",
                return_value={
                    "loss_classifier": loss_classifier,
                    "loss_box_reg": loss_box_reg,
                },
            ),
            patch.object(trainer.model, "train"),
            patch.object(trainer.model, "to", return_value=trainer.model),
        ):
            # Act - should not raise
            trainer.train(device, train_loader, val_loader)

    def test_build_model_returns_faster_rcnn(self, trainer):
        """Test that build_model returns a FasterRCNN model."""
        # Act
        model = trainer.build_model()

        # Assert
        assert model is not None
        assert hasattr(model, "roi_heads")
        assert hasattr(model.roi_heads, "box_predictor")
