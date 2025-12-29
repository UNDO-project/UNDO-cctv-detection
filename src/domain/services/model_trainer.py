from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader


class ModelTrainer(ABC):
    """Abstract base class for model trainers.

    Supports both DataLoader-based training (PyTorch models like Faster R-CNN)
    and config-file-based training (YOLO/Ultralytics).
    """

    @abstractmethod
    def train(
        self,
        device: torch.device,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        data_config: Path | None = None,
    ) -> None:
        """Train the model on the provided data.

        Either (train_loader AND val_loader) or data_config must be provided.
        PyTorch models use loaders, YOLO models use data_config.

        :param device: PyTorch device for training (cpu, cuda, mps)
        :param train_loader: Optional training dataloader for PyTorch models
        :param val_loader: Optional validation dataloader for PyTorch models
        :param data_config: Optional path to YAML config file for YOLO models
        :return: None
        :rtype: None
        :raises ValueError: If neither (train_loader, val_loader) nor data_config provided
        """
        pass

    @staticmethod
    def validate_inputs(
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        data_config: Path | None,
    ) -> None:
        """Validate that required inputs are provided.

        :param train_loader: Training dataloader
        :param val_loader: Validation dataloader
        :param data_config: Path to data config file
        :return: None
        :rtype: None
        :raises ValueError: If inputs are invalid
        """
        has_loaders = train_loader is not None and val_loader is not None
        has_config = data_config is not None

        if not (has_loaders or has_config):
            msg = "Either (train_loader AND val_loader) or data_config must be provided"
            raise ValueError(msg)
