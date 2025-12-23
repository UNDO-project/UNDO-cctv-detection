"""Unit tests for DistanceCalculator domain service."""

import pytest
import math
from src.domain.services.distance_calculator import DistanceCalculator, DoriLevel


class TestDoriLevel:
    """Test DoriLevel enum."""

    def test_dori_level_values(self):
        """Test that DoriLevel enum has correct PPM values."""
        assert DoriLevel.DETECTION.value == 25
        assert DoriLevel.OBSERVATION.value == 63
        assert DoriLevel.RECOGNITION.value == 125
        assert DoriLevel.IDENTIFICATION.value == 250

    def test_dori_level_order(self):
        """Test that DoriLevel values are in ascending order."""
        levels = [
            DoriLevel.DETECTION.value,
            DoriLevel.OBSERVATION.value,
            DoriLevel.RECOGNITION.value,
            DoriLevel.IDENTIFICATION.value,
        ]
        assert levels == sorted(levels), "DORI levels should be in ascending order"


class TestCalculateMaxDistanceDori:
    """Test calculate_max_distance_dori method."""

    def test_recognition_level_standard_person(self):
        """Test max distance calculation for person recognition (125 PPM)."""
        result = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080, target_height_m=1.7, ppm=DoriLevel.RECOGNITION.value
        )
        expected = (1080 * 1.7) / 125
        assert result == pytest.approx(expected, rel=0.01)
        assert result == pytest.approx(14.688, rel=0.01)

    def test_detection_level(self):
        """Test max distance calculation for detection (25 PPM)."""
        result = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080, target_height_m=1.7, ppm=DoriLevel.DETECTION.value
        )
        expected = (1080 * 1.7) / 25
        assert result == pytest.approx(expected, rel=0.01)
        assert result == pytest.approx(73.44, rel=0.01)

    def test_observation_level(self):
        """Test max distance calculation for observation (63 PPM)."""
        result = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080, target_height_m=1.7, ppm=DoriLevel.OBSERVATION.value
        )
        expected = (1080 * 1.7) / 63
        assert result == pytest.approx(expected, rel=0.01)
        assert result == pytest.approx(29.143, rel=0.01)

    def test_identification_level(self):
        """Test max distance calculation for identification (250 PPM)."""
        result = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080,
            target_height_m=1.7,
            ppm=DoriLevel.IDENTIFICATION.value,
        )
        expected = (1080 * 1.7) / 250
        assert result == pytest.approx(expected, rel=0.01)
        assert result == pytest.approx(7.344, rel=0.01)

    def test_higher_resolution_increases_distance(self):
        """Test that higher resolution allows greater distance."""
        distance_1080p = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080, target_height_m=1.7, ppm=DoriLevel.RECOGNITION.value
        )
        distance_4k = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=2160, target_height_m=1.7, ppm=DoriLevel.RECOGNITION.value
        )
        assert distance_4k == pytest.approx(2 * distance_1080p, rel=0.01)

    def test_taller_target_increases_distance(self):
        """Test that taller targets allow greater distance."""
        distance_person = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080, target_height_m=1.7, ppm=DoriLevel.RECOGNITION.value
        )
        distance_vehicle = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080, target_height_m=3.0, ppm=DoriLevel.RECOGNITION.value
        )
        assert distance_vehicle > distance_person

    def test_higher_ppm_decreases_distance(self):
        """Test that higher PPM requirement decreases max distance."""
        distance_detection = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080, target_height_m=1.7, ppm=DoriLevel.DETECTION.value
        )
        distance_identification = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080,
            target_height_m=1.7,
            ppm=DoriLevel.IDENTIFICATION.value,
        )
        assert distance_detection > distance_identification

    def test_with_custom_ppm_value(self):
        """Test calculation with custom PPM value."""
        result = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1920, target_height_m=2.0, ppm=100
        )
        expected = (1920 * 2.0) / 100
        assert result == pytest.approx(expected, rel=0.01)
        assert result == pytest.approx(38.4, rel=0.01)

    def test_returns_float(self):
        """Test that result is always a float."""
        result = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080, target_height_m=1.7, ppm=125
        )
        assert isinstance(result, float)


class TestCalculateDistanceFov:
    """Test calculate_distance_fov method."""

    def test_standard_fov_calculation(self):
        """Test distance calculation with standard FOV."""
        result = DistanceCalculator.calculate_distance_fov(
            target_width_m=2.0, hfov_deg=75.0
        )
        expected = 2.0 / (2 * math.tan(math.radians(75.0 / 2)))
        assert result == pytest.approx(expected, rel=0.01)
        assert result > 0

    def test_narrow_fov_increases_distance(self):
        """Test that narrower FOV increases distance."""
        distance_wide = DistanceCalculator.calculate_distance_fov(
            target_width_m=2.0, hfov_deg=90.0
        )
        distance_narrow = DistanceCalculator.calculate_distance_fov(
            target_width_m=2.0, hfov_deg=45.0
        )
        assert distance_narrow > distance_wide

    def test_wider_target_increases_distance(self):
        """Test that wider targets increase distance."""
        distance_small = DistanceCalculator.calculate_distance_fov(
            target_width_m=1.0, hfov_deg=75.0
        )
        distance_large = DistanceCalculator.calculate_distance_fov(
            target_width_m=3.0, hfov_deg=75.0
        )
        assert distance_large > distance_small
        assert distance_large == pytest.approx(3 * distance_small, rel=0.01)

    def test_very_narrow_fov(self):
        """Test calculation with very narrow FOV (telephoto lens)."""
        result = DistanceCalculator.calculate_distance_fov(
            target_width_m=2.0, hfov_deg=10.0
        )
        assert result > 10.0  # Should be at significant distance

    def test_very_wide_fov(self):
        """Test calculation with very wide FOV (fisheye lens)."""
        result = DistanceCalculator.calculate_distance_fov(
            target_width_m=2.0, hfov_deg=120.0
        )
        assert result > 0
        assert result < 2.0  # Wide FOV means closer distance

    def test_90_degree_fov(self):
        """Test calculation with 90 degree FOV (common security camera)."""
        result = DistanceCalculator.calculate_distance_fov(
            target_width_m=2.0, hfov_deg=90.0
        )
        # For 90° FOV: 2.0 / (2 * tan(45°)) = 2.0 / (2 * 1.0) = 1.0
        expected = 2.0 / (2 * math.tan(math.radians(45.0)))
        assert result == pytest.approx(expected, rel=0.01)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_returns_float(self):
        """Test that result is always a float."""
        result = DistanceCalculator.calculate_distance_fov(
            target_width_m=2.0, hfov_deg=75.0
        )
        assert isinstance(result, float)

    def test_symmetry_of_calculation(self):
        """Test that doubling both width and angle maintains proportional distance."""
        distance1 = DistanceCalculator.calculate_distance_fov(
            target_width_m=1.0, hfov_deg=60.0
        )
        distance2 = DistanceCalculator.calculate_distance_fov(
            target_width_m=2.0, hfov_deg=60.0
        )
        # Distance should double when width doubles (for same FOV)
        assert distance2 == pytest.approx(2 * distance1, rel=0.01)


class TestDistanceCalculatorIntegration:
    """Integration tests combining both calculation methods."""

    def test_both_methods_return_positive_values(self):
        """Test that both methods return positive distances."""
        dori_result = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080, target_height_m=1.7, ppm=125
        )
        fov_result = DistanceCalculator.calculate_distance_fov(
            target_width_m=2.0, hfov_deg=75.0
        )

        assert dori_result > 0
        assert fov_result > 0

    def test_realistic_surveillance_scenario(self):
        """Test a realistic CCTV surveillance scenario."""
        # 1080p camera, recognizing a person at reasonable distance
        dori_distance = DistanceCalculator.calculate_max_distance_dori(
            sensor_height_px=1080,
            target_height_m=1.7,  # Average person height
            ppm=DoriLevel.RECOGNITION.value,
        )

        # Same camera with 75-degree FOV covering a 3m wide area
        fov_distance = DistanceCalculator.calculate_distance_fov(
            target_width_m=3.0, hfov_deg=75.0
        )

        # Both should give reasonable surveillance distances (5-20m typically)
        assert 5.0 < dori_distance < 20.0
        assert 1.0 < fov_distance < 10.0
