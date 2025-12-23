from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch.utils.data import Dataset


class DatasetSplitter(ABC):
    @abstractmethod
    def split(
        self, dataset: Any, train_ratio: float, val_ratio: float
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split the dataset into train, validation, and test subsets.
        :param dataset: The provided dataset
        :param train_ratio: The ratio of the training dataset
        :param val_ratio: The ratio of the validation dataset
        :return: Training data, validation data and test data
        """
        pass
