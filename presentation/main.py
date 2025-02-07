from application.camera_image_downloader import CameraImageDownloader
from infrastructure.data_loaders import CameraDataLoader
from infrastructure.image_scraper import ImageScraper
from config import OUTPUT_DIR, CSV_FILE


def main():
    data_loader = CameraDataLoader()
    image_scraper = ImageScraper(output_dir=OUTPUT_DIR, headless=False)
    downloader = CameraImageDownloader(data_loader, image_scraper)

    downloader.download_images(csv_path=CSV_FILE)


if __name__ == "__main__":
    main()


"""
def main():
    # Instantiate the domain service
    distance_calculator = DistanceCalculator()
    
    # Create the application service
    service = SurveillanceService(distance_calculator)
    
    # Example 1: DORI-based recognition distance
    sensor_height_px = 1080  # e.g., Full HD sensor height
    target_height_m = 1.7    # Average human height in meters
    max_recognition_distance = service.get_max_recognition_distance(sensor_height_px, target_height_m)
    logger.info(f"Maximum recognition distance: {max_recognition_distance:.2f} meters")
    
    # Example 2: FoV-based observation distance
    target_width_m = 2       # e.g., approximate width of a car in meters
    hfov_deg = 75            # Example horizontal field of view
    max_observation_distance = service.get_max_observation_distance(target_width_m, hfov_deg)
    logger.info(f"Maximum observation distance: {max_observation_distance:.2f} meters")

if __name__ == "__main__":
    main()
"""
