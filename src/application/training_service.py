from typing import Any

from torch.utils.data import DataLoader

from src.config import settings
from src.domain.services.data_splitter import DatasetSplitter
from src.domain.services.model_trainer import ModelTrainer
from src.infrastructure.device_selector import DeviceSelector


class TrainingService:
    """Service for orchestrating model training workflow.

    This service coordinates dataset splitting, data loading, and model training
    using configurable hyperparameters from application settings.
    """

    def __init__(
        self,
        dataset: Any,
        model_trainer: ModelTrainer,
        dataset_splitter: DatasetSplitter,
    ) -> None:
        """Initialize training service.

        :param dataset: Dataset to train on
        :param model_trainer: Model trainer implementation
        :param dataset_splitter: Dataset splitter implementation
        """
        self.dataset = dataset
        self.model_trainer = model_trainer
        self.dataset_splitter = dataset_splitter

    def run_training(
        self,
        train_ratio: float = settings.training.train_ratio,
        val_ratio: float = settings.training.val_ratio,
        batch_size: int = settings.training.batch_size,
    ) -> None:
        """Run the training workflow.

        :param train_ratio: Training set ratio (default from settings)
        :param val_ratio: Validation set ratio (default from settings)
        :param batch_size: Batch size for data loaders (default from settings)
        """
        train_data, val_data, _test_data = self.dataset_splitter.split(
            self.dataset, train_ratio, val_ratio
        )
        train_loader = DataLoader(
            train_data,
            batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x, strict=True)),
        )
        val_loader = DataLoader(
            val_data,
            batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x, strict=True)),
        )

        # Note: test_data is available but not currently used in training workflow.
        # Future enhancement: Add post-training evaluation using test set.

        device = DeviceSelector.get_optimal_device()

        # Pass device first, then loaders (new unified interface)
        # YOLO trainers will ignore loaders and use their data_config
        # PyTorch trainers (Faster R-CNN) will use the loaders
        self.model_trainer.train(device, train_loader, val_loader)
