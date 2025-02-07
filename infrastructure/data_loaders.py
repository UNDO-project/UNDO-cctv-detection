from pathlib import Path
from typing import List
import pandas as pd

from domain.camera import CameraDataFromCsv


class CameraDataLoader:
    def load_camera_data(self, csv_path: Path) -> List[CameraDataFromCsv]:
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
