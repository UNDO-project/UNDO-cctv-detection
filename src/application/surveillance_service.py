from src.domain.services.distance_calculator import DistanceCalculator, DoriLevel


class SurveillanceService:
    def __init__(self, distance_calculator: DistanceCalculator):
        self.distance_calculator = distance_calculator

    def get_max_recognition_distance(
        self, sensor_height_px: int, target_height_px: float
    ) -> float:
        """
        Calculate the maximum distance for recognition-level surveillance using the DORI standard.
        This uses the recognition level (125 PPM).
        :param sensor_height_px: The sensor height in pixels
        :param target_height_px: The target height in pixels
        :return: The maximum recognition distance
        """
        return self.distance_calculator.calculate_max_distance_dori(
            sensor_height_px, target_height_px, DoriLevel.RECOGNITION.value
        )

    def get_max_observation_distance(
        self, target_width_m: float, hfov_deg: float
    ) -> float:
        """
        Calculate the maximum observation distance based on the horizontal field of view.
        :param target_width_m: The width of the target in meters
        :param hfov_deg: Horizontal field of view in degrees
        :return: The maximum observation distance
        """
        return self.distance_calculator.calculate_distance_fov(target_width_m, hfov_deg)
