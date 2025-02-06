from pathlib import Path
from loguru import logger

from src.config import CSV_FILE, OUTPUT_DIR
from src.data_loaders import load_camera_data
from src.image_scrapers import capture_screenshots


def image_downloader(csv_path: Path, output_dir: Path):
    """
    Loads camera data from a csv, visits links, takes screenshots and persists in dedicated folder.
    :param csv_path: The Path object to csv file.
    :param output_dir: The Path object to output directory.
    :return:
    """
    cameras = load_camera_data(csv_path)
    logger.info(f"Urls found: {len(cameras)}")
    capture_screenshots(cameras, output_dir)


if __name__ == "__main__":
    image_downloader(csv_path=CSV_FILE, output_dir=OUTPUT_DIR)
