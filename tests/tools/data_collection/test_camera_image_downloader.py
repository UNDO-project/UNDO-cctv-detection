"""Unit tests for CameraImageDownloader.

The downloader loads cameras from a CSV and hands them to the scraper; the
single test below pins that wiring.
"""

from pathlib import Path
from unittest.mock import Mock

from src.domain.camera import CameraDataFromCsv
from src.tools.data_collection.camera_image_downloader import CameraImageDownloader


class TestCameraImageDownloader:
    """Test CameraImageDownloader orchestration."""

    def test_download_images_scrapes_cameras_loaded_from_csv(self):
        """Cameras loaded from the CSV path are passed to the scraper."""
        cameras = [
            CameraDataFromCsv(62.2426, 25.7473, "https://example.com/camera1"),
            CameraDataFromCsv(60.1699, 24.9384, "https://example.com/camera2"),
        ]
        data_loader = Mock()
        data_loader.load_camera_data.return_value = cameras
        image_scraper = Mock()
        csv_path = Path("/data/cameras.csv")

        CameraImageDownloader(data_loader, image_scraper).download_images(csv_path)

        data_loader.load_camera_data.assert_called_once_with(csv_path)
        image_scraper.scrape_images.assert_called_once_with(cameras)
