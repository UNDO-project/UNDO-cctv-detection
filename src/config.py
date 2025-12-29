from pathlib import Path
import os
from dotenv import load_dotenv

# Project root is parent of src/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ------------------- Directory Structure -------------------#
SRC_DIR: Path = PROJECT_ROOT / "src"
DATASETS: Path = PROJECT_ROOT / "datasets"
SAMPLES: Path = PROJECT_ROOT / "samples"
DOCS: Path = PROJECT_ROOT / "docs"

# ------------------- Dataset Paths -------------------#
IMAGES_DIR: Path = DATASETS / "images"
LABELS_DIR: Path = DATASETS / "labels"
ULTRALYTICS_DIR: Path = DATASETS / "ultralytics"

# ------------------- Model Weights -------------------#
YOLO_WEIGHTS: Path = SAMPLES / "best.pt"
FASTER_RCNN_WEIGHTS: Path = SAMPLES / "fasterrcnn_best.pt"
DETR_WEIGHTS: Path = SAMPLES / "detr_best.pt"

# ------------------- Training Config -------------------#
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.3
BATCH_SIZE: int = 4
DATA_CONFIG: Path = PROJECT_ROOT / "data.yaml"

# ------------------- Scraper Config -------------------#
CSV_FILE: Path = DATASETS / "cctv-aware-jyvaskyla.csv"
OUTPUT_DIR: Path = DATASETS / "screenshots"
REJECT_ALL: str = "Reject all"
REJECT_ALL_GR: str = "Απόρριψη όλων"

# ------------------- Environment Overrides -------------------#

load_dotenv()

# Allow environment variables to override defaults
if override := os.getenv("CCTV_MODEL_WEIGHTS"):
    YOLO_WEIGHTS = Path(override)

if override := os.getenv("CCTV_DATASET_DIR"):
    DATASETS = Path(override)
    IMAGES_DIR = DATASETS / "images"
    LABELS_DIR = DATASETS / "labels"
