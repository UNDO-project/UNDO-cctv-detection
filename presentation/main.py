from pathlib import Path

import torch.optim
from loguru import logger

from config import PROJECT_ROOT
from infrastructure.trainers import YoloUltralyticsTrainer


def main():
    # Get data.yaml from project root
    data_config = str(Path(PROJECT_ROOT / "data.yaml"))

    model_weights = "yolov8n.pt"

    epochs = 20
    img_size = 640

    model_trainer = YoloUltralyticsTrainer(model_weights, data_config, epochs, img_size)

    # For Apple M2, choose a device (this is optional since Ultralytics does its own device handling).
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logger.info(f"Starting training on device: {device}")
    # The train() method’s DataLoader parameters are not used by the Ultralytics trainer.
    model_trainer.train(None, None, device)


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


"""
def main():
    data_loader = CameraDataLoader()
    image_scraper = ImageScraper(output_dir=OUTPUT_DIR, headless=False)
    downloader = CameraImageDownloader(data_loader, image_scraper)

    downloader.download_images(csv_path=CSV_FILE)
"""

"""
def main():
    # Load the dataset:
    images = IMAGES_DIR
    annotations = LABELS_DIR
    dataset = CCTVDetectionDatasetLoader(images, annotations)

    # Build the model
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    # Instantiate trainer
    model_trainer = YoloTrainer(model, optimizer, num_epochs)

    # Instantiate dataset splitter
    dataset_splitter = SklearnDatasetSplitter()

    # Create training service
    training_service = TrainingService(dataset, model_trainer, dataset_splitter)

    # Run training
    training_service.run_training(train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, batch_size=BATCH_SIZE)
    
"""
