"""Unit tests for CameraImageDownloader application service.

This module tests the CameraImageDownloader which orchestrates downloading
camera images by loading CSV data and scraping images from URLs.
"""

import pytest
from unittest.mock import Mock
from pathlib import Path

from tools.data_collection.camera_image_downloader import CameraImageDownloader
from src.domain.camera import CameraDataFromCsv


class TestCameraImageDownloader:
    """Test CameraImageDownloader application service."""

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock CameraDataLoader.

        :return: Mock CameraDataLoader instance
        """
        return Mock()

    @pytest.fixture
    def mock_image_scraper(self):
        """Create a mock ImageScraper.

        :return: Mock ImageScraper instance
        """
        return Mock()

    @pytest.fixture
    def service(self, mock_data_loader, mock_image_scraper):
        """Create CameraImageDownloader with mocked dependencies.

        :param mock_data_loader: Mocked data loader
        :param mock_image_scraper: Mocked image scraper
        :return: CameraImageDownloader instance
        """
        return CameraImageDownloader(mock_data_loader, mock_image_scraper)

    @pytest.fixture
    def sample_cameras(self):
        """Create sample camera data.

        :return: List of CameraDataFromCsv objects
        """
        return [
            CameraDataFromCsv(62.2426, 25.7473, "https://example.com/camera1"),
            CameraDataFromCsv(60.1699, 24.9384, "https://example.com/camera2"),
            CameraDataFromCsv(61.4978, 23.7610, "https://example.com/camera3"),
        ]

    def test_initialization(self, mock_data_loader, mock_image_scraper):
        """Test that service initializes with dependencies.

        :param mock_data_loader: Mocked data loader
        :param mock_image_scraper: Mocked image scraper
        :return: None
        """
        service = CameraImageDownloader(mock_data_loader, mock_image_scraper)

        assert service.data_loader is mock_data_loader
        assert service.image_scraper is mock_image_scraper

    def test_download_images_loads_camera_data(
        self, service, mock_data_loader, sample_cameras
    ):
        """Test that download_images loads camera data from CSV.

        :param service: CameraImageDownloader instance
        :param mock_data_loader: Mocked data loader
        :param sample_cameras: Sample camera data
        :return: None
        """
        mock_data_loader.load_camera_data.return_value = sample_cameras
        csv_path = Path("/data/cameras.csv")

        service.download_images(csv_path)

        mock_data_loader.load_camera_data.assert_called_once_with(csv_path)

    def test_download_images_scrapes_loaded_cameras(
        self, service, mock_data_loader, mock_image_scraper, sample_cameras
    ):
        """Test that download_images passes cameras to scraper.

        :param service: CameraImageDownloader instance
        :param mock_data_loader: Mocked data loader
        :param mock_image_scraper: Mocked image scraper
        :param sample_cameras: Sample camera data
        :return: None
        """
        mock_data_loader.load_camera_data.return_value = sample_cameras

        service.download_images(Path("/data/cameras.csv"))

        mock_image_scraper.scrape_images.assert_called_once_with(sample_cameras)

    def test_download_images_execution_order(
        self, service, mock_data_loader, mock_image_scraper, sample_cameras
    ):
        """Test that loading happens before scraping.

        :param service: CameraImageDownloader instance
        :param mock_data_loader: Mocked data loader
        :param mock_image_scraper: Mocked image scraper
        :param sample_cameras: Sample camera data
        :return: None
        """
        execution_order = []

        def track_load(*args, **kwargs):
            execution_order.append("load")
            return sample_cameras

        def track_scrape(*args, **kwargs):
            execution_order.append("scrape")

        mock_data_loader.load_camera_data.side_effect = track_load
        mock_image_scraper.scrape_images.side_effect = track_scrape

        service.download_images(Path("/data/cameras.csv"))

        assert execution_order == ["load", "scrape"]

    def test_download_images_with_empty_camera_list(
        self, service, mock_data_loader, mock_image_scraper
    ):
        """Test behavior when no cameras are found.

        :param service: CameraImageDownloader instance
        :param mock_data_loader: Mocked data loader
        :param mock_image_scraper: Mocked image scraper
        :return: None
        """
        mock_data_loader.load_camera_data.return_value = []

        service.download_images(Path("/data/cameras.csv"))

        # Should still call scraper with empty list
        mock_image_scraper.scrape_images.assert_called_once_with([])

    def test_download_images_with_single_camera(
        self, service, mock_data_loader, mock_image_scraper
    ):
        """Test downloading images for a single camera.

        :param service: CameraImageDownloader instance
        :param mock_data_loader: Mocked data loader
        :param mock_image_scraper: Mocked image scraper
        :return: None
        """
        single_camera = [
            CameraDataFromCsv(62.2426, 25.7473, "https://example.com/camera1")
        ]
        mock_data_loader.load_camera_data.return_value = single_camera

        service.download_images(Path("/data/single.csv"))

        mock_image_scraper.scrape_images.assert_called_once_with(single_camera)

    def test_download_images_returns_none(
        self, service, mock_data_loader, sample_cameras
    ):
        """Test that download_images returns None.

        :param service: CameraImageDownloader instance
        :param mock_data_loader: Mocked data loader
        :param sample_cameras: Sample camera data
        :return: None
        """
        mock_data_loader.load_camera_data.return_value = sample_cameras

        result = service.download_images(Path("/data/cameras.csv"))

        assert result is None

    def test_download_images_with_different_csv_paths(
        self, service, mock_data_loader, sample_cameras
    ):
        """Test downloading with various CSV paths.

        :param service: CameraImageDownloader instance
        :param mock_data_loader: Mocked data loader
        :param sample_cameras: Sample camera data
        :return: None
        """
        mock_data_loader.load_camera_data.return_value = sample_cameras

        csv_paths = [
            Path("/data/cameras.csv"),
            Path("relative/path/cameras.csv"),
            Path("/home/user/data/cameras.csv"),
        ]

        for csv_path in csv_paths:
            mock_data_loader.reset_mock()
            service.download_images(csv_path)

            mock_data_loader.load_camera_data.assert_called_once_with(csv_path)
