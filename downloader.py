from pathlib import Path
from loguru import logger

from src.config import CSV_FILE, OUTPUT_DIR
from src.data_loaders import load_camera_data
from src.image_scrapers import capture_screenshots


def image_downloader(csv_path: Path, output_dir: Path):
    cameras = load_camera_data(csv_path)
    logger.info(f"Urls found: {len(cameras)}")
    capture_screenshots(cameras, output_dir)


if __name__ == "__main__":
    image_downloader(csv_path=CSV_FILE, output_dir=OUTPUT_DIR)
