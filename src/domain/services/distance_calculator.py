from enum import Enum
import math
from src.domain.exceptions import ValidationError


class DoriLevel(Enum):
    DETECTION = 25  # Pixels per Meter for detection
    OBSERVATION = 63  # Pixels per Meter for observation
    RECOGNITION = 125  # Pixels per Meter for recognition
    IDENTIFICATION = 250  # Pixels per Meter for identification


class DistanceCalculator:
    """
            A domain service that encapsulates calculations related to camera surveillance.
            It provides methods to compute the maximum distance at which a camera can capture an
            optimal image based on the DORI standard and field-of-view (FoV) considerations.

    DORI standard (Detection, Observation, Recognition, Identification) and the Focal Length & Field of View (FoV) equation.
    Determines the maximum distance between the camera lens and the target within which the camera captures an optimal image
    for a given surveillance purpose.

    DORI standard (IEC 62676-4):

    -------------------|-----------------------|----------------------------------------------|
    Surveillance Level	Pixels per Meter (PPM)	Typical Use Case
    -------------------|-----------------------|----------------------------------------------|
    Detection	        25 PPM	                Detecting the presence of a person/vehicle    |
    Observation	        63 PPM	                Determining behavior and general actions      |
    Recognition	        125 PPM	                Recognizing a person’s face or license plate  |
    Identification	    250 PPM	                Confirming identity with high certainty       |
    -------------------|-----------------------|----------------------------------------------|
    """

    @staticmethod
    def calculate_max_distance_dori(
        sensor_height_px: int, target_height_m: float, ppm: int
    ) -> float:
        """
        Calculate the maximum distance at which a camera can capture an optimal image based on the Dori standard
        :param sensor_height_px: Camera sensor height in pixels
        :param target_height_m: Real-world height of the target in meters
        :param ppm: Required pixels per meter (DORI standard)
        :return: Maximum distance in meters
        """
        if sensor_height_px <= 0:
            raise ValidationError(
                f"sensor_height_px must be positive, got {sensor_height_px}"
            )
        if target_height_m <= 0:
            raise ValidationError(
                f"target_height_m must be positive, got {target_height_m}"
            )
        if ppm <= 0:
            raise ValidationError(f"ppm must be positive, got {ppm}")
        return (sensor_height_px * target_height_m) / ppm

    @staticmethod
    def calculate_distance_fov(target_width_m: float, hfov_deg: float) -> float:
        """
        Calculate the maximum distance based on field of view.
        :param target_width_m: Width of the target in meters
        :param hfov_deg: Horizontal field of view in degrees
        :return: Maximum observation distance in meters
        """
        if target_width_m <= 0:
            raise ValidationError(
                f"target_width_m must be positive, got {target_width_m}"
            )
        if not (0 < hfov_deg < 180):
            raise ValidationError(f"hfov_deg must be in range (0, 180), got {hfov_deg}")
        return target_width_m / (2 * math.tan(math.radians(hfov_deg / 2)))
