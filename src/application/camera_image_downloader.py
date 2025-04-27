from pathlib import Path

from loguru import logger

from src.infrastructure.data_loaders import CameraDataLoader
from src.infrastructure.image_scraper import ImageScraper


class CameraImageDownloader:
    def __init__(self, data_loader: CameraDataLoader, image_scraper: ImageScraper):
        self.data_loader = data_loader
        self.image_scraper = image_scraper

    def download_images(self, csv_path: Path) -> None:
        """
        Loads camera data from a csv, visits links, takes screenshots and persists in dedicated folder.
        :param csv_path: The Path object to csv file.
        :return:
        """
        cameras = self.data_loader.load_camera_data(csv_path)
        logger.info(f"Urls found: {len(cameras)}")
        self.image_scraper.scrape_images(cameras)
