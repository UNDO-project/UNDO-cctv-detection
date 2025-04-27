import shutil
from pathlib import Path
from loguru import logger

from src.domain.services.dataset_preparer import DatasetPreparer
from src.infrastructure.splitters import SklearnDatasetSplitter


class SklearnDatasetPreparer(DatasetPreparer):
    def prepare_ultralytics_dataset(
        self,
        source_images: Path,
        source_labels: Path,
        output_images: Path,
        output_labels: Path,
        train_ratio: float,
        val_ratio: float,
        move_files: bool = True,
    ) -> None:
        # Gather all image files
        image_files = list(source_images.glob("*.*"))
        if not image_files:
            logger.warning(f"No images found in {source_images}.")
            return

        # Build list of (image, label) tuples, warn if a label is missing
        dataset = []
        for img in image_files:
            label_file = source_labels / f"{img.stem}.txt"
            if not label_file.exists():
                logger.warning(f"Missing label for image {img.name}; skipping.")
                continue
            dataset.append((img, label_file))

        if not dataset:
            logger.error(
                "No valid image-label pairs found. Aborting dataset preparation."
            )
            return

        # Split the dataset
        splitter = SklearnDatasetSplitter()
        train_data, val_data, test_data = splitter.split(
            dataset, train_ratio, val_ratio
        )
        splits = {
            "train": train_data,
            "val": val_data,
            "test": test_data,  # You might not use test in training
        }

        # Organize files into the split directories
        for split_name, data in splits.items():
            if not data:
                logger.info(f"No data for split: {split_name}.")
                continue

            split_img_dir = output_images / split_name
            split_lbl_dir = output_labels / split_name
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file, lbl_file in data:
                dest_img = split_img_dir / img_file.name
                dest_lbl = split_lbl_dir / lbl_file.name
                if move_files:
                    shutil.move(img_file, dest_img)
                    shutil.move(lbl_file, dest_lbl)
                else:
                    shutil.copy(img_file, dest_img)
                    shutil.copy(lbl_file, dest_lbl)

            logger.info(
                f"Prepared {len(data)} items for '{split_name}' split at "
                f"{split_img_dir} and {split_lbl_dir}."
            )
        logger.info("Dataset preparation completed successfully.")
