"""Unit tests for CameraDataFromCsv domain entity.

This module tests the CameraDataFromCsv dataclass which represents
camera location and URL data extracted from CSV files.
"""

import pytest
from src.domain.camera import CameraDataFromCsv


class TestCameraDataFromCsv:
    """Test CameraDataFromCsv domain entity."""

    def test_create_camera_with_valid_data(self):
        """Test creating a camera entity with valid data.

        :return: None
        """
        camera = CameraDataFromCsv(
            latitude=62.2426, longitude=25.7473, url="https://example.com/camera1"
        )

        assert camera.latitude == 62.2426
        assert camera.longitude == 25.7473
        assert camera.url == "https://example.com/camera1"

    def test_camera_is_frozen(self):
        """Test that CameraDataFromCsv is immutable (frozen).

        :return: None
        """
        camera = CameraDataFromCsv(
            latitude=62.2426, longitude=25.7473, url="https://example.com/camera1"
        )

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            camera.latitude = 100.0

    def test_camera_equality(self):
        """Test that cameras with same data are equal.

        :return: None
        """
        camera1 = CameraDataFromCsv(
            latitude=62.2426, longitude=25.7473, url="https://example.com/camera1"
        )
        camera2 = CameraDataFromCsv(
            latitude=62.2426, longitude=25.7473, url="https://example.com/camera1"
        )

        assert camera1 == camera2

    def test_camera_inequality(self):
        """Test that cameras with different data are not equal.

        :return: None
        """
        camera1 = CameraDataFromCsv(
            latitude=62.2426, longitude=25.7473, url="https://example.com/camera1"
        )
        camera2 = CameraDataFromCsv(
            latitude=62.2426,
            longitude=25.7473,
            url="https://example.com/camera2",  # Different URL
        )

        assert camera1 != camera2

    def test_camera_with_negative_coordinates(self):
        """Test creating camera with negative coordinates.

        :return: None
        """
        camera = CameraDataFromCsv(
            latitude=-34.6037,
            longitude=-58.3816,
            url="https://example.com/camera_south",
        )

        assert camera.latitude == -34.6037
        assert camera.longitude == -58.3816

    def test_camera_repr(self):
        """Test string representation of camera.

        :return: None
        """
        camera = CameraDataFromCsv(
            latitude=62.2426, longitude=25.7473, url="https://example.com/camera1"
        )

        repr_str = repr(camera)
        assert "CameraDataFromCsv" in repr_str
        assert "62.2426" in repr_str
        assert "25.7473" in repr_str
        assert "https://example.com/camera1" in repr_str

    def test_camera_hash(self):
        """Test that frozen dataclass is hashable.

        :return: None
        """
        camera = CameraDataFromCsv(
            latitude=62.2426, longitude=25.7473, url="https://example.com/camera1"
        )

        # Should be able to add to set (requires hashable)
        camera_set = {camera}
        assert camera in camera_set

    def test_multiple_cameras_in_collection(self):
        """Test storing multiple cameras in a collection.

        :return: None
        """
        cameras = [
            CameraDataFromCsv(62.2426, 25.7473, "https://example.com/camera1"),
            CameraDataFromCsv(60.1699, 24.9384, "https://example.com/camera2"),
            CameraDataFromCsv(61.4978, 23.7610, "https://example.com/camera3"),
        ]

        assert len(cameras) == 3
        assert all(isinstance(cam, CameraDataFromCsv) for cam in cameras)
