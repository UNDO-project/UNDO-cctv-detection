"""Unit tests for SurveillanceService application service."""

import pytest
from unittest.mock import Mock
from src.application.surveillance_service import SurveillanceService
from src.domain.services.distance_calculator import DoriLevel


class TestSurveillanceService:
    """Test SurveillanceService application service."""

    @pytest.fixture
    def mock_calculator(self):
        """Create a mock DistanceCalculator."""
        return Mock()

    @pytest.fixture
    def service(self, mock_calculator):
        """Create SurveillanceService with mocked calculator."""
        return SurveillanceService(mock_calculator)

    def test_initialization(self, mock_calculator):
        """Test that service initializes with calculator."""
        service = SurveillanceService(mock_calculator)
        assert service.distance_calculator is mock_calculator

    def test_get_max_recognition_distance_delegates_to_calculator(
        self, service, mock_calculator
    ):
        """Test that get_max_recognition_distance delegates to calculator."""
        # Arrange
        mock_calculator.calculate_max_distance_dori.return_value = 15.0
        sensor_height = 1080
        target_height = 1.7

        # Act
        result = service.get_max_recognition_distance(sensor_height, target_height)

        # Assert
        assert result == 15.0
        mock_calculator.calculate_max_distance_dori.assert_called_once_with(
            sensor_height, target_height, DoriLevel.RECOGNITION.value
        )

    def test_get_max_recognition_distance_uses_recognition_level(
        self, service, mock_calculator
    ):
        """Test that recognition distance uses RECOGNITION PPM level (125)."""
        # Arrange
        mock_calculator.calculate_max_distance_dori.return_value = 15.0

        # Act
        service.get_max_recognition_distance(1080, 1.7)

        # Assert
        # Verify it was called with RECOGNITION level (125 PPM)
        call_args = mock_calculator.calculate_max_distance_dori.call_args
        assert call_args[0][2] == DoriLevel.RECOGNITION.value
        assert call_args[0][2] == 125

    def test_get_max_recognition_distance_returns_calculator_result(
        self, service, mock_calculator
    ):
        """Test that service returns the calculator's result."""
        # Arrange
        expected_distance = 22.5
        mock_calculator.calculate_max_distance_dori.return_value = expected_distance

        # Act
        result = service.get_max_recognition_distance(1920, 1.8)

        # Assert
        assert result == expected_distance

    def test_get_max_observation_distance_delegates_to_calculator(
        self, service, mock_calculator
    ):
        """Test that get_max_observation_distance delegates to calculator."""
        # Arrange
        mock_calculator.calculate_distance_fov.return_value = 10.5
        target_width = 3.0
        hfov = 75.0

        # Act
        result = service.get_max_observation_distance(target_width, hfov)

        # Assert
        assert result == 10.5
        mock_calculator.calculate_distance_fov.assert_called_once_with(
            target_width, hfov
        )

    def test_get_max_observation_distance_returns_calculator_result(
        self, service, mock_calculator
    ):
        """Test that service returns the calculator's FOV result."""
        # Arrange
        expected_distance = 8.3
        mock_calculator.calculate_distance_fov.return_value = expected_distance

        # Act
        result = service.get_max_observation_distance(2.5, 60.0)

        # Assert
        assert result == expected_distance

    def test_multiple_recognition_calls(self, service, mock_calculator):
        """Test that service can handle multiple recognition calls."""
        # Arrange
        mock_calculator.calculate_max_distance_dori.side_effect = [10.0, 15.0, 20.0]

        # Act
        result1 = service.get_max_recognition_distance(720, 1.7)
        result2 = service.get_max_recognition_distance(1080, 1.7)
        result3 = service.get_max_recognition_distance(1920, 1.7)

        # Assert
        assert result1 == 10.0
        assert result2 == 15.0
        assert result3 == 20.0
        assert mock_calculator.calculate_max_distance_dori.call_count == 3

    def test_multiple_observation_calls(self, service, mock_calculator):
        """Test that service can handle multiple observation calls."""
        # Arrange
        mock_calculator.calculate_distance_fov.side_effect = [5.0, 7.5, 12.0]

        # Act
        result1 = service.get_max_observation_distance(2.0, 90.0)
        result2 = service.get_max_observation_distance(3.0, 75.0)
        result3 = service.get_max_observation_distance(4.0, 60.0)

        # Assert
        assert result1 == 5.0
        assert result2 == 7.5
        assert result3 == 12.0
        assert mock_calculator.calculate_distance_fov.call_count == 3

    def test_both_methods_can_be_called_independently(self, service, mock_calculator):
        """Test that both service methods work independently."""
        # Arrange
        mock_calculator.calculate_max_distance_dori.return_value = 15.0
        mock_calculator.calculate_distance_fov.return_value = 8.0

        # Act
        recognition_result = service.get_max_recognition_distance(1080, 1.7)
        observation_result = service.get_max_observation_distance(3.0, 75.0)

        # Assert
        assert recognition_result == 15.0
        assert observation_result == 8.0
        mock_calculator.calculate_max_distance_dori.assert_called_once()
        mock_calculator.calculate_distance_fov.assert_called_once()


class TestSurveillanceServiceIntegration:
    """Integration tests using real DistanceCalculator."""

    @pytest.fixture
    def service_with_real_calculator(self):
        """Create SurveillanceService with real DistanceCalculator."""
        from src.domain.services.distance_calculator import DistanceCalculator

        return SurveillanceService(DistanceCalculator)

    def test_recognition_distance_with_real_calculator(
        self, service_with_real_calculator
    ):
        """Test recognition distance calculation with real calculator."""
        result = service_with_real_calculator.get_max_recognition_distance(1080, 1.7)

        # Should match the DORI calculation: (1080 * 1.7) / 125
        assert result == pytest.approx(14.688, rel=0.01)
        assert result > 0

    def test_observation_distance_with_real_calculator(
        self, service_with_real_calculator
    ):
        """Test observation distance calculation with real calculator."""
        result = service_with_real_calculator.get_max_observation_distance(3.0, 75.0)

        # Should return a positive distance
        assert result > 0
        assert isinstance(result, float)

    def test_realistic_surveillance_scenario_with_real_calculator(
        self, service_with_real_calculator
    ):
        """Test a realistic CCTV scenario with real calculator."""
        # 1080p camera recognizing a person
        recognition_dist = service_with_real_calculator.get_max_recognition_distance(
            sensor_height_px=1080, target_height_px=1.7
        )

        # Same camera covering a 3m wide entrance with 75° FOV
        observation_dist = service_with_real_calculator.get_max_observation_distance(
            target_width_m=3.0, hfov_deg=75.0
        )

        # Both should give reasonable distances
        assert 10.0 < recognition_dist < 20.0
        assert 1.5 < observation_dist < 10.0

    def test_higher_resolution_increases_recognition_distance(
        self, service_with_real_calculator
    ):
        """Test that higher resolution increases recognition distance."""
        distance_1080p = service_with_real_calculator.get_max_recognition_distance(
            1080, 1.7
        )
        distance_4k = service_with_real_calculator.get_max_recognition_distance(
            2160, 1.7
        )

        assert distance_4k > distance_1080p
        assert distance_4k == pytest.approx(2 * distance_1080p, rel=0.01)
