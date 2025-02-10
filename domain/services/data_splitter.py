from abc import ABC, abstractmethod


class DataSplitter(ABC):
    @abstractmethod
    def split(self, dataset, train_ratio: float, val_ratio: float):
        """
        Split the dataset into train, validation, and test subsets.
        :param dataset: The provided dataset
        :param train_ratio: The ratio of the training dataset
        :param val_ratio: The ratio of the validation dataset
        :return: Training data, validation data and test data
        """
        pass
