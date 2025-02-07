from abc import ABC, abstractmethod
from pathlib import Path


class ImageConverter(ABC):
    @abstractmethod
    def convert_heic_to_jpg(self, input_folder: Path, output_folder: Path) -> None:
        """
        Convert all HEIC images in the input_folder to JPG format, saving them into the output_folder.
        :param input_folder: The Path object representing the input folder
        :param output_folder: The Path object representing the output folder
        :return:
        """
        pass
