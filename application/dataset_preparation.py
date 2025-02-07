from pathlib import Path

from domain.services.image_converter import ImageConverter


class DatasetPreparation:
    def __init__(self, image_converter: ImageConverter):
        self.image_converter = image_converter

    def prepare_dateset(self, input_folder: Path, output_folder: Path) -> None:
        """
        Prepare the dataset by converting all HEIC images to JPG format.
        :param input_folder: The Path object representing input folder
        :param output_folder: The Path object representing output folder
        :return:
        """
        self.image_converter.convert_heic_to_jpg(input_folder, output_folder)
