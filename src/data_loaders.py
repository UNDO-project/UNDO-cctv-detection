from pathlib import Path
from typing import List, Tuple
import pandas as pd


def load_camera_data(csv_path: Path) -> List[Tuple[float, float, str]]:
    """Load camera data from a CSV file and extract valid unique URLs with coordinates."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["url"])  # Remove rows where URL is missing
    df = df.drop_duplicates(subset=["url"])  # Keep only unique URLs

    cameras = [
        (row["latitude"], row["longitude"], row["url"]) for _, row in df.iterrows()
    ]
    return cameras
