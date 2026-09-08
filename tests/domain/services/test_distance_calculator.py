"""Unit tests for DistanceCalculator domain service.

Expected values are hand-computed oracles rather than restatements of the
formula, so a wrong implementation cannot make them pass.
"""

import pytest

from src.domain.exceptions import ValidationError
from src.domain.services.distance_calculator import DistanceCalculator, DoriLevel


class TestDoriLevel:
    """Test DoriLevel enum."""

    def test_dori_level_values_follow_iec_62676_4(self):
        """Pixel-per-metre values match the IEC 62676-4 DORI standard."""
        assert DoriLevel.DETECTION.value == 25
        assert DoriLevel.OBSERVATION.value == 63
        assert DoriLevel.RECOGNITION.value == 125
        assert DoriLevel.IDENTIFICATION.value == 250


class TestCalculateMaxDistanceDori:
    """Test calculate_max_distance_dori method."""

    @pytest.mark.parametrize(
        ("sensor_height_px", "target_height_m", "ppm", "expected_m"),
        [
            (1080, 1.7, DoriLevel.DETECTION.value, 73.44),
            (1080, 1.7, DoriLevel.OBSERVATION.value, 29.143),
            (1080, 1.7, DoriLevel.RECOGNITION.value, 14.688),
            (1080, 1.7, DoriLevel.IDENTIFICATION.value, 7.344),
            (2160, 1.7, DoriLevel.RECOGNITION.value, 29.376),
            (1920, 2.0, 100, 38.4),
        ],
    )
    def test_max_distance_for_each_dori_level(
        self, sensor_height_px, target_height_m, ppm, expected_m
    ):
        """Max distance is (sensor_px * target_m) / ppm for every DORI level."""
        result = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=sensor_height_px,
            target_height_m=target_height_m,
            ppm=ppm,
        )
        assert result == pytest.approx(expected_m, rel=1e-3)


class TestCalculateDistanceFov:
    """Test calculate_distance_fov method."""

    @pytest.mark.parametrize(
        ("target_width_m", "hfov_deg", "expected_m"),
        [
            (2.0, 90.0, 1.0),  # tan(45 deg) = 1
            (2.0, 75.0, 1.3032),
            (3.0, 75.0, 1.9548),
            (2.0, 10.0, 11.430),  # telephoto
            (2.0, 120.0, 0.5774),  # fisheye
        ],
    )
    def test_distance_for_given_width_and_fov(
        self, target_width_m, hfov_deg, expected_m
    ):
        """Distance is width / (2 * tan(hfov / 2))."""
        result = DistanceCalculator.calculate_distance_fov(
            target_width_m=target_width_m, hfov_deg=hfov_deg
        )
        assert result == pytest.approx(expected_m, rel=1e-3)

    def test_narrower_fov_increases_distance(self):
        """Halving the field of view pushes the distance out."""
        wide = DistanceCalculator.calculate_distance_fov(2.0, 90.0)
        narrow = DistanceCalculator.calculate_distance_fov(2.0, 45.0)
        assert narrow > wide


class TestInputValidation:
    """Test that invalid inputs are rejected with ValidationError."""

    @pytest.mark.parametrize(
        ("sensor_height_px", "target_height_m", "ppm", "field"),
        [
            (0, 1.7, 125, "sensor_height_px"),
            (-1080, 1.7, 125, "sensor_height_px"),
            (1080, 0.0, 125, "target_height_m"),
            (1080, -1.7, 125, "target_height_m"),
            (1080, 1.7, 0, "ppm"),
            (1080, 1.7, -25, "ppm"),
        ],
    )
    def test_dori_rejects_non_positive_inputs(
        self, sensor_height_px, target_height_m, ppm, field
    ):
        """Each DORI argument must be strictly positive; the message names the field."""
        with pytest.raises(ValidationError, match=field):
            DistanceCalculator.calculate_max_distance_dori(
                sensor_height_px=sensor_height_px,
                target_height_m=target_height_m,
                ppm=ppm,
            )

    @pytest.mark.parametrize(
        ("target_width_m", "hfov_deg", "field"),
        [
            (0.0, 90.0, "target_width_m"),
            (-1.0, 90.0, "target_width_m"),
            (1.0, 0.0, "hfov_deg"),
            (1.0, -10.0, "hfov_deg"),
            (1.0, 180.0, "hfov_deg"),
            (1.0, 200.0, "hfov_deg"),
        ],
    )
    def test_fov_rejects_out_of_range_inputs(self, target_width_m, hfov_deg, field):
        """Width must be positive and the FOV strictly inside (0, 180) degrees."""
        with pytest.raises(ValidationError, match=field):
            DistanceCalculator.calculate_distance_fov(
                target_width_m=target_width_m, hfov_deg=hfov_deg
            )

    def test_fov_accepts_boundary_neighbours(self):
        """Values just inside the FOV range are accepted."""
        assert DistanceCalculator.calculate_distance_fov(1.0, 0.001) > 0
        assert DistanceCalculator.calculate_distance_fov(1.0, 179.999) > 0
