from pathlib import Path
from typing import List, Tuple
import pandas as pd


def load_camera_data(csv_path: Path) -> List[Tuple[float, float, str]]:
    """Load camera data from a CSV file and extract valid URLs with coordinates."""
    df = pd.read_csv(csv_path)
    cameras = []
    for _, row in df.iterrows():
        if pd.notna(row["url"]):  # Ensure the URL is not missing
            cameras.append((row["latitude"], row["longitude"], row["url"]))
    return cameras
