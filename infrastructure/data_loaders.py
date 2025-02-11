from pathlib import Path
from typing import List
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToTensor

from domain.camera import CameraDataFromCsv


class CameraDataLoader:
    @staticmethod
    def load_camera_data(csv_path: Path) -> List[CameraDataFromCsv]:
        """
        Load camera data from a CSV file and extract valid unique URLs with coordinates.
        :param csv_path: The Path object to the csv file.
        :return: A list of camera information: lon, lat, url.
        """
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["url"])  # Remove rows where URL is missing
        df = df.drop_duplicates(subset=["url"])  # Keep only unique URLs

        cameras = [
            CameraDataFromCsv(row["latitude"], row["longitude"], row["url"])
            for _, row in df.iterrows()
        ]
        return cameras


class CCTVDetectionDatasetLoader(Dataset):
    """
    A custom dataset for loading CCTV camera images and their YOLO-formatted annotations.
    Each annotation file (a .txt file with the same base name as the image) contains lines formatted as:
      class x_center y_center width height
    where the coordinates are normalized with respect to the image dimensions.
    """

    def __init__(self, images: Path, annotations: Path, transforms=None):
        self.images = images
        self.annotations = annotations
        self.transforms = transforms
        self.image_files = sorted([p.name for p in images.iterdir() if p.is_file()])

        if self.transforms is None:
            self.transforms = lambda image, target: (ToTensor()(image), target)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        image_file = self.image_files[idx]
        image_path = self.images / image_file  # using pathlib
        image = Image.open(image_path).convert("RGB")
        width_img, height_img = image.size

        # Build the path to the annotation file
        base_name = image_path.stem
        annotation_file = base_name + ".txt"
        annotation_path = self.annotations / annotation_file

        boxes = []
        labels = []
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                x_center, y_center, box_width, box_height = map(float, parts[1:])
                xmin = (x_center - box_width / 2) * width_img
                ymin = (y_center - box_height / 2) * height_img
                xmax = (x_center + box_width / 2) * width_img
                ymax = (y_center + box_height / 2) * height_img
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls)

        # Ensure the boxes tensor is of shape [N, 4]
        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)

        # Similarly, ensure labels tensor is of shape [N]
        if labels:
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            labels_tensor = torch.empty((0,), dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
