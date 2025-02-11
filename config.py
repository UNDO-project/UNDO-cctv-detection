from pathlib import Path

# ------------------- Training config -------------------#
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATASETS: Path = PROJECT_ROOT / "datasets"
IMAGES_DIR: Path = DATASETS / "images"
LABELS_DIR: Path = DATASETS / "labels"
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.3
BATCH_SIZE: int = 4

# ------------------- Scraper config -------------------#
CSV_FILE: Path = Path("../../datasets/cctv-aware-jyvaskyla.csv")
OUTPUT_DIR: Path = Path(
    "../datasets/screenshots"
)  # Folder to save screenshots of images
REJECT_ALL: str = "Reject all"
REJECT_ALL_GR: str = "Απόρριψη όλων"
