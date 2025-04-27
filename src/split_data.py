import shutil

from pathlib import Path
from typing import List

import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import IMAGES_DIR, PROJECT_ROOT, LABELS_DIR, TRAIN_RATIO


def split_dataset(
    source_images_dir: Path,
    source_labels_dir: Path,
    train_ratio: float,
    output_dir: Path,
) -> (Path, Path):
    """
    Splits the dataset images into training and validation subsets.
    :param source_images_dir: Directory containing all original images.
    :param source_labels_dir: Directory containing all original label files.
    :param train_ratio: Fraction of images to use for training.
    :param output_dir: The base directory where the new train/val folders will be created.
    :return: The directories for training images and validation images.
    """
    # List all image files in the source directory with particular image extensions
    image_files = sorted(
        [
            p.name
            for p in source_images_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )
    train_files, val_files = train_test_split(
        image_files, train_size=train_ratio, random_state=42
    )

    # Create the new directory structure:
    images_train_dir = output_dir / "images" / "train"
    images_val_dir = output_dir / "images" / "val"
    labels_train_dir = output_dir / "labels" / "train"
    labels_val_dir = output_dir / "labels" / "val"

    for directory in [
        images_train_dir,
        images_val_dir,
        labels_train_dir,
        labels_val_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copy training files
    for filename in train_files:
        # Copy image
        src_img = source_images_dir / filename
        dst_img = images_train_dir / filename
        shutil.copy(src_img, dst_img)

        # Copy corresponding label and change extension to .txt
        label_filename = Path(filename).with_suffix(".txt")
        src_label = source_labels_dir / label_filename
        if src_label.exists():
            dst_label = labels_train_dir / label_filename
            shutil.copy(src_label, dst_label)

    # Copy validation files
    for filename in val_files:
        src_img = source_images_dir / filename
        dst_img = images_val_dir / filename
        shutil.copy(src_img, dst_img)

        label_filename = Path(filename).with_suffix(".txt")
        src_label = source_labels_dir / label_filename
        if src_label.exists():
            dst_label = labels_val_dir / label_filename
            shutil.copy(src_label, dst_label)

    logger.info(
        f"Split complete: {len(train_files)} train images, {len(val_files)} val images."
    )
    return images_train_dir, images_val_dir


def create_data_yaml(
    train_img_dir: Path, val_img_dir: Path, nc: int, names: List, output_yaml_path: Path
) -> None:
    """
    Creates a YAML file for Ultralytics YOLO that specifies the training and validation directories.
    :param train_img_dir: Directory of training images.
    :param val_img_dir: Directory of validation images.
    :param nc: Number of classes.
    :param names: List of class names.
    :param output_yaml_path: Path where the YAML file will be written.
    :return:
    """
    data_config = {
        "train": str(train_img_dir.resolve()),
        "val": str(val_img_dir.resolve()),
        "nc": nc,
        "names": names,
    }

    with open(output_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    logger.info(f"data.yaml created at: {output_yaml_path.resolve()}")


if __name__ == "__main__":
    # Determine the project root based on this script's location.
    project_root = PROJECT_ROOT

    # Define source directories (adjust these if your structure is different)
    source_images_dir = IMAGES_DIR
    source_labels_dir = LABELS_DIR

    # Output directory is the same "datasets" directory (subdirectories will be created inside it)
    output_dir = project_root / "datasets" / "ultralytics"

    # Set the desired train ratio
    train_ratio = TRAIN_RATIO

    # Split the dataset
    train_images_dir, val_images_dir = split_dataset(
        source_images_dir, source_labels_dir, train_ratio, output_dir
    )

    # Define dataset metadata: number of classes and class names.
    nc = 2
    names = ["CCTV", "CCTV-SIGNS"]

    # Create the data.yaml file in the project root.
    output_yaml_path = project_root / "data.yaml"
    create_data_yaml(train_images_dir, val_images_dir, nc, names, output_yaml_path)
