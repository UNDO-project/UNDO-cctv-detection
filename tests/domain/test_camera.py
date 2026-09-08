"""Unit tests for the CameraDataFromCsv domain entity.

The entity is a plain frozen dataclass; immutability is the only design
decision worth pinning.
"""

from dataclasses import FrozenInstanceError

import pytest

from src.domain.camera import CameraDataFromCsv


class TestCameraDataFromCsv:
    """Test CameraDataFromCsv value object."""

    def test_camera_is_frozen(self):
        """Camera records cannot be mutated after construction."""
        camera = CameraDataFromCsv(62.2426, 25.7473, "https://example.com/camera")

        with pytest.raises(FrozenInstanceError):
            camera.latitude = 100.0
