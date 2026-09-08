"""Unit tests for DETR model trainer.

Tests the DETRTrainer implementation to ensure it correctly implements the
ModelTrainer interface and properly initializes DETR models for training.
"""

from pathlib import Path
from types import SimpleNamespace
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


class TestCollateFn:
    """collate_fn pads a batch of variable-sized images and builds pixel masks."""

    def test_pads_to_largest_image_in_batch(self):
        """All images are zero-padded to the max height and width in the batch."""
        small = torch.ones(3, 4, 6)
        large = torch.ones(3, 8, 5)
        batch = [
            {"pixel_values": small, "labels": {"class_labels": torch.tensor([0])}},
            {"pixel_values": large, "labels": {"class_labels": torch.tensor([1])}},
        ]

        out = DETRTrainer.collate_fn(batch)

        assert out["pixel_values"].shape == (2, 3, 8, 6)
        assert out["pixel_mask"].shape == (2, 8, 6)
        # Original content is preserved in the top-left corner
        assert torch.equal(out["pixel_values"][0, :, :4, :6], small)
        assert torch.equal(out["pixel_values"][1, :, :8, :5], large)
        # Everything outside the original extent is zero padding
        assert out["pixel_values"][0, :, 4:, :].sum() == 0
        assert out["pixel_values"][0, :, :, 6:].sum() == 0
        assert out["pixel_values"][1, :, :, 5:].sum() == 0

    def test_pixel_mask_marks_real_pixels_only(self):
        """Mask is 1 over the real image extent and 0 over padding."""
        batch = [
            {"pixel_values": torch.ones(3, 2, 3), "labels": {}},
            {"pixel_values": torch.ones(3, 4, 4), "labels": {}},
        ]

        mask = DETRTrainer.collate_fn(batch)["pixel_mask"]

        assert mask.dtype == torch.long
        assert mask[0].sum() == 2 * 3
        assert torch.equal(mask[0, :2, :3], torch.ones(2, 3, dtype=torch.long))
        assert mask[0, 2:, :].sum() == 0
        assert mask[0, :, 3:].sum() == 0
        assert mask[1].sum() == 4 * 4

    def test_labels_are_kept_as_list_and_dtype_preserved(self):
        """Labels are passed through untouched; pixel dtype is not changed."""
        labels = [
            {"class_labels": torch.tensor([0, 1])},
            {"class_labels": torch.tensor([])},
        ]
        batch = [
            {
                "pixel_values": torch.ones(3, 2, 2, dtype=torch.float16),
                "labels": labels[0],
            },
            {
                "pixel_values": torch.ones(3, 2, 2, dtype=torch.float16),
                "labels": labels[1],
            },
        ]

        out = DETRTrainer.collate_fn(batch)

        assert out["labels"] is not None
        assert out["labels"][0] is labels[0]
        assert out["labels"][1] is labels[1]
        assert out["pixel_values"].dtype == torch.float16


class TestEvaluateMap:
    """evaluate_map converts normalized centre boxes to COCO and scores them."""

    @staticmethod
    def _trainer_with_outputs(logits: torch.Tensor, pred_boxes: torch.Tensor):
        """Build a DETRTrainer without loading weights, with a stubbed model.

        :param logits: (num_queries, num_labels + 1) class logits for one image
        :param pred_boxes: (num_queries, 4) normalized (cx, cy, w, h) boxes
        :return: Trainer whose model returns the given outputs
        :rtype: DETRTrainer
        """
        trainer = DETRTrainer.__new__(DETRTrainer)
        trainer.num_labels = 2
        outputs = SimpleNamespace(
            logits=logits.unsqueeze(0), pred_boxes=pred_boxes.unsqueeze(0)
        )
        trainer.model = MagicMock(return_value=outputs)
        return trainer

    @staticmethod
    def _single_image_batch():
        """One 100x200 image with a single class-0 box at the centre."""
        return {
            "pixel_values": torch.zeros(1, 3, 100, 200),
            "pixel_mask": torch.ones(1, 100, 200, dtype=torch.long),
            "labels": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.4]]),
                    "class_labels": torch.tensor([0]),
                }
            ],
        }

    def test_perfect_prediction_scores_full_map(self):
        """A prediction matching the ground truth box exactly yields mAP 1.0."""
        # Query 0 confidently predicts class 0; the rest predict "no object"
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
        pred_boxes = torch.tensor(
            [[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]
        )
        trainer = self._trainer_with_outputs(logits, pred_boxes)

        result = trainer.evaluate_map([self._single_image_batch()], torch.device("cpu"))

        assert result["map50"] == pytest.approx(1.0)
        assert result["map"] == pytest.approx(1.0)

    def test_no_confident_queries_yields_zero_map(self):
        """When every query is below the 0.05 score threshold, mAP is 0."""
        logits = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
        pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.5, 0.5, 0.2, 0.4]])
        trainer = self._trainer_with_outputs(logits, pred_boxes)

        result = trainer.evaluate_map([self._single_image_batch()], torch.device("cpu"))

        assert result == {"map50": 0.0, "map": 0.0}

    def test_misplaced_prediction_scores_zero_at_iou_50(self):
        """A confident box with no overlap with ground truth gives mAP@0.5 of 0."""
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        pred_boxes = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        trainer = self._trainer_with_outputs(logits, pred_boxes)

        result = trainer.evaluate_map([self._single_image_batch()], torch.device("cpu"))

        assert result["map50"] == pytest.approx(0.0)
