"""Unit tests for DETR model trainer.

Tests the DETRTrainer implementation to ensure it correctly implements the
ModelTrainer interface and properly initializes DETR models for training.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.infrastructure.detr_trainer import DETRTrainer


class TestDETRTrainer:
    """Test cases for DETRTrainer class."""

    def test_initialization(self):
        """Test that DETR trainer initializes correctly with default parameters."""
        with (
            patch("src.infrastructure.detr_trainer.DetrImageProcessor"),
            patch("src.infrastructure.detr_trainer.DetrForObjectDetection"),
        ):
            trainer = DETRTrainer(num_labels=2, epochs=1)
            assert trainer.num_labels == 2
            assert trainer.epochs == 1
            assert trainer.model is not None
            assert trainer.processor is not None

    def test_initialization_with_custom_params(self):
        """Test that DETR trainer accepts custom hyperparameters."""
        with (
            patch("src.infrastructure.detr_trainer.DetrImageProcessor"),
            patch("src.infrastructure.detr_trainer.DetrForObjectDetection"),
        ):
            trainer = DETRTrainer(
                model_name="facebook/detr-resnet-50",
                num_labels=3,
                epochs=10,
                learning_rate=2e-4,
                batch_size=8,
            )
            assert trainer.num_labels == 3
            assert trainer.epochs == 10
            assert trainer.learning_rate == 2e-4
            assert trainer.batch_size == 8

    def test_requires_dataloaders(self):
        """Test that training fails when dataloaders are not provided."""
        with (
            patch("src.infrastructure.detr_trainer.DetrImageProcessor"),
            patch("src.infrastructure.detr_trainer.DetrForObjectDetection"),
        ):
            trainer = DETRTrainer(num_labels=2, epochs=1)

            with pytest.raises(ValueError, match="train_loader"):
                trainer.train(device=torch.device("cpu"))

    def test_requires_both_train_and_val_loaders(self):
        """Test that both train and validation loaders are required."""
        with (
            patch("src.infrastructure.detr_trainer.DetrImageProcessor"),
            patch("src.infrastructure.detr_trainer.DetrForObjectDetection"),
        ):
            trainer = DETRTrainer(num_labels=2, epochs=1)

            # Create a dummy dataloader
            dummy_data = TensorDataset(torch.randn(10, 3, 224, 224))
            train_loader = DataLoader(dummy_data, batch_size=2)

            with pytest.raises(ValueError, match="train_loader"):
                trainer.train(device=torch.device("cpu"), train_loader=train_loader)

    def test_train_with_mocked_components(self):
        """Test training with mocked HuggingFace components."""
        with (
            patch("src.infrastructure.detr_trainer.DetrImageProcessor"),
            patch("src.infrastructure.detr_trainer.DetrForObjectDetection"),
            patch("src.infrastructure.detr_trainer.Trainer") as mock_trainer_class,
        ):
            # Setup mock trainer
            mock_trainer_instance = MagicMock()
            mock_trainer_class.return_value = mock_trainer_instance

            # Create trainer
            trainer = DETRTrainer(num_labels=2, epochs=1, batch_size=2)

            # Create dummy dataloaders
            dummy_data = TensorDataset(torch.randn(10, 3, 224, 224))
            train_loader = DataLoader(dummy_data, batch_size=2)
            val_loader = DataLoader(dummy_data, batch_size=2)

            # Train
            trainer.train(
                device=torch.device("cpu"),
                train_loader=train_loader,
                val_loader=val_loader,
            )

            # Verify trainer was called
            mock_trainer_class.assert_called_once()
            mock_trainer_instance.train.assert_called_once()
            mock_trainer_instance.save_model.assert_called_once()

    def test_save_weights(self, tmp_path: Path):
        """Test that model weights can be saved."""
        with (
            patch("src.infrastructure.detr_trainer.DetrImageProcessor"),
            patch("src.infrastructure.detr_trainer.DetrForObjectDetection"),
            patch("torch.save") as mock_save,
        ):
            trainer = DETRTrainer(num_labels=2, epochs=1)

            weights_path = tmp_path / "test_weights.pt"
            trainer.save_weights(weights_path)

            mock_save.assert_called_once()
            assert mock_save.call_args[0][1] == weights_path

    def test_output_dir_creation(self, tmp_path: Path):
        """Test that custom output directory is used."""
        with (
            patch("src.infrastructure.detr_trainer.DetrImageProcessor"),
            patch("src.infrastructure.detr_trainer.DetrForObjectDetection"),
        ):
            custom_output = tmp_path / "custom_detr_output"
            trainer = DETRTrainer(num_labels=2, epochs=1, output_dir=custom_output)

            assert trainer.output_dir == custom_output

    def test_model_name_configuration(self):
        """Test that different DETR model variants can be specified."""
        with (
            patch("src.infrastructure.detr_trainer.DetrImageProcessor"),
            patch(
                "src.infrastructure.detr_trainer.DetrForObjectDetection"
            ) as mock_model,
        ):
            custom_model = "facebook/detr-resnet-101"
            trainer = DETRTrainer(model_name=custom_model, num_labels=2, epochs=1)

            assert trainer.model_name == custom_model
            mock_model.from_pretrained.assert_called_once()
            call_args = mock_model.from_pretrained.call_args
            assert call_args[0][0] == custom_model
