from abc import ABC, abstractmethod
from typing import Any


class ModelTrainer(ABC):
    @abstractmethod
    def train(self, train_loader: Any, val_loader: Any, device: Any) -> None:
        """
        Train the model using the training and validation data loaders on the specified device.
        :param train_loader: Training dataloader
        :param val_loader: Validation dataloader
        :param device: 'cpu', 'gpu' or 'mps' depending on system
        :return:
        """
        pass
