"""Integration tests for CameraDataLoader.

This module tests the CameraDataLoader which loads camera location
and URL data from CSV files.
"""

import pytest
import pandas as pd

from src.infrastructure.data_loaders import CameraDataLoader
from src.domain.camera import CameraDataFromCsv


class TestCameraDataLoader:
    """Test CameraDataLoader infrastructure implementation."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file with camera data.

        :param tmp_path: Temporary directory path
        :return: Path to created CSV file
        """
        csv_path = tmp_path / "cameras.csv"
        data = {
            "latitude": [62.2426, 60.1699, 61.4978],
            "longitude": [25.7473, 24.9384, 23.7610],
            "url": [
                "https://example.com/camera1",
                "https://example.com/camera2",
                "https://example.com/camera3",
            ],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def csv_with_nulls(self, tmp_path):
        """Create CSV file with null URLs.

        :param tmp_path: Temporary directory path
        :return: Path to created CSV file
        """
        csv_path = tmp_path / "cameras_with_nulls.csv"
        data = {
            "latitude": [62.2426, 60.1699, 61.4978, 59.9139],
            "longitude": [25.7473, 24.9384, 23.7610, 10.7522],
            "url": [
                "https://example.com/camera1",
                None,  # Null URL
                "https://example.com/camera3",
                "https://example.com/camera4",
            ],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def csv_with_duplicates(self, tmp_path):
        """Create CSV file with duplicate URLs.

        :param tmp_path: Temporary directory path
        :return: Path to created CSV file
        """
        csv_path = tmp_path / "cameras_with_duplicates.csv"
        data = {
            "latitude": [62.2426, 60.1699, 62.2426],
            "longitude": [25.7473, 24.9384, 25.7473],
            "url": [
                "https://example.com/camera1",
                "https://example.com/camera2",
                "https://example.com/camera1",  # Duplicate
            ],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_load_camera_data_returns_list(self, sample_csv):
        """Test that load_camera_data returns a list.

        :param sample_csv: Sample CSV file path
        :return: None
        """
        result = CameraDataLoader.load_camera_data(sample_csv)
        assert isinstance(result, list)

    def test_load_camera_data_returns_camera_objects(self, sample_csv):
        """Test that loaded data contains CameraDataFromCsv objects.

        :param sample_csv: Sample CSV file path
        :return: None
        """
        result = CameraDataLoader.load_camera_data(sample_csv)
        assert all(isinstance(cam, CameraDataFromCsv) for cam in result)

    def test_load_camera_data_correct_count(self, sample_csv):
        """Test that correct number of cameras are loaded.

        :param sample_csv: Sample CSV file path
        :return: None
        """
        result = CameraDataLoader.load_camera_data(sample_csv)
        assert len(result) == 3

    def test_load_camera_data_correct_values(self, sample_csv):
        """Test that camera data values are loaded correctly.

        :param sample_csv: Sample CSV file path
        :return: None
        """
        result = CameraDataLoader.load_camera_data(sample_csv)

        assert result[0].latitude == 62.2426
        assert result[0].longitude == 25.7473
        assert result[0].url == "https://example.com/camera1"

    def test_load_camera_data_removes_null_urls(self, csv_with_nulls):
        """Test that rows with null URLs are removed.

        :param csv_with_nulls: CSV file with null URLs
        :return: None
        """
        result = CameraDataLoader.load_camera_data(csv_with_nulls)

        # Should have 3 cameras (4 total minus 1 with null URL)
        assert len(result) == 3
        # All URLs should be non-null
        assert all(cam.url is not None for cam in result)

    def test_load_camera_data_removes_duplicate_urls(self, csv_with_duplicates):
        """Test that duplicate URLs are removed.

        :param csv_with_duplicates: CSV file with duplicate URLs
        :return: None
        """
        result = CameraDataLoader.load_camera_data(csv_with_duplicates)

        # Should have 2 cameras (3 total minus 1 duplicate)
        assert len(result) == 2

        # URLs should be unique
        urls = [cam.url for cam in result]
        assert len(urls) == len(set(urls))

    def test_load_camera_data_with_empty_csv(self, tmp_path):
        """Test loading from empty CSV.

        :param tmp_path: Temporary directory path
        :return: None
        """
        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame(columns=["latitude", "longitude", "url"])
        df.to_csv(csv_path, index=False)

        result = CameraDataLoader.load_camera_data(csv_path)
        assert result == []

    def test_load_camera_data_preserves_order(self, sample_csv):
        """Test that camera order from CSV is preserved.

        :param sample_csv: Sample CSV file path
        :return: None
        """
        result = CameraDataLoader.load_camera_data(sample_csv)

        expected_urls = [
            "https://example.com/camera1",
            "https://example.com/camera2",
            "https://example.com/camera3",
        ]

        actual_urls = [cam.url for cam in result]
        assert actual_urls == expected_urls

    def test_load_camera_data_handles_negative_coordinates(self, tmp_path):
        """Test loading cameras with negative coordinates.

        :param tmp_path: Temporary directory path
        :return: None
        """
        csv_path = tmp_path / "negative_coords.csv"
        data = {
            "latitude": [-34.6037, -33.8688],
            "longitude": [-58.3816, 151.2093],
            "url": ["https://example.com/camera1", "https://example.com/camera2"],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        result = CameraDataLoader.load_camera_data(csv_path)

        assert len(result) == 2
        assert result[0].latitude == -34.6037
        assert result[0].longitude == -58.3816
