from pathlib import Path
from PIL import Image
from loguru import logger
import pillow_heif

from domain.services.image_converter import ImageConverter


class PillowImageConverter(ImageConverter):
    def convert_heic_to_jpg(self, input_folder: Path, output_folder: Path):
        """
        Convert all HEIC images in a folder to JPG format.
        :param input_folder: Path to input folder
        :param output_folder: Path to output folder
        :return:
        """
        # Ensure that output folder exists
        output_folder.mkdir(parents=True, exist_ok=True)

        for heic_file in input_folder.glob("*HEIC"):
            try:
                # Read the heic file
                heif_file = pillow_heif.read_heif(heic_file)
                image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
                output_path = output_folder / heic_file.with_suffix(".jpg").name

                image.save(output_path, format="JPEG")
                logger.info(f"Converted: {heic_file} -> {output_path}")
            except Exception as e:
                logger.error(f"Failed to convert {heic_file.name}: {e}")
                raise e
