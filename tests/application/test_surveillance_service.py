"""Tests for SurveillanceService using the real DistanceCalculator.

The service is a thin delegation layer. The only logic worth checking is
that each method wires the correct DORI level and formula into the
calculator, which is only observable with the real implementation.
"""

import pytest

from src.application.surveillance_service import SurveillanceService
from src.domain.services.distance_calculator import DistanceCalculator


class TestSurveillanceService:
    """Test SurveillanceService against the real calculator."""

    @pytest.fixture
    def service(self):
        """Create SurveillanceService with the real DistanceCalculator."""
        return SurveillanceService(DistanceCalculator)

    def test_recognition_distance_uses_recognition_ppm(self, service):
        """Recognition distance is (sensor_px * target_m) / 125 PPM."""
        assert service.get_max_recognition_distance(1080, 1.7) == pytest.approx(14.688)

    def test_observation_distance_uses_fov_formula(self, service):
        """Observation distance is width / (2 * tan(hfov / 2))."""
        # 3.0 / (2 * tan(37.5 degrees))
        assert service.get_max_observation_distance(3.0, 75.0) == pytest.approx(
            1.9548, rel=1e-3
        )
