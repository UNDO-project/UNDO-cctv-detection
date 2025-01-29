import math

from loguru import logger

"""
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


def calculate_max_distance(
    sensor_height_px: int, target_height_m: float, ppm: int
) -> float:
    """
    Calculate the maximum distance at which a camera can capture an optimal image.
    :param sensor_height_px: Camera sensor height in pixels
    :param target_height_m: Real-world height of the target in meters
    :param ppm: Required pixels per meter (DORI standard)
    :return: Maximum distance in meters
    """
    return (sensor_height_px * target_height_m) / ppm


def calculate_distance_fov(target_width_m: float, hfov_deg: float) -> float:
    """
    Calculate the maximum distance based on field of view.
    :param target_width_m: Width of the target in meters
    :param hfov_deg: Horizontal field of view in degrees
    :return: Maximum observation distance in meters
    """
    return target_width_m / (2 * math.tan(math.radians(hfov_deg / 2)))


if __name__ == "__main__":
    # Example 1: DORI Calculation
    # sensor_height_px = 1080  # Full HD camera
    # target_height_m = 1.7  # Human height
    # ppm = 125  # Recognition level (DORI)
    max_distance_dori = calculate_max_distance(
        sensor_height_px=1080, target_height_m=1.7, ppm=125
    )
    logger.info(f"Maximum recognition distance: {max_distance_dori:.2f} meters")

    # Example 2: HFOV Calculation
    # target_width_m = 2  # Approximate width of a car
    # hfov_deg = 75  # Example field of view
    max_distance_fov = calculate_distance_fov(target_width_m=2, hfov_deg=75)
    logger.info(f"Maximum observation distance: {max_distance_fov:.2f} meters")
