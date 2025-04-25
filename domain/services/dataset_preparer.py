from abc import ABC, abstractmethod
from pathlib import Path


class DatasetPreparer(ABC):
    @abstractmethod
    def prepare_ultralytics_dataset(
        self,
        source_images: Path,
        source_labels: Path,
        output_images: Path,
        output_labels: Path,
        train_ratio: float,
        val_ratio: float,
        move_files: bool = True,
    ) -> None:
        """
        Prepare and split the dataset for Ultralytics training.

        :param source_images: Folder with the original images.
        :param source_labels: Folder with the original labels.
        :param output_images: Base folder where split images will be stored.
        :param output_labels: Base folder where split labels will be stored.
        :param train_ratio: Fraction of data for training.
        :param val_ratio: Fraction of data for validation.
        :param move_files: If True, move files; if False, copy files.
        """
        pass
