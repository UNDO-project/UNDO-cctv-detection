# ⚙️ Configuration Guide

This document provides comprehensive information on configuring the CCTV Detection System using environment variables, `.env` files, and Pydantic settings.

## Overview

The project uses **Pydantic v2 Settings** for type-safe, validated configuration with automatic environment variable support. All configuration is centralized in `src/config.py`.

## Configuration Structure

### Configuration Classes

#### `PathsConfig`
Manages project paths with computed fields.

```python
class PathsConfig(BaseModel):
    project_root: Path          # Auto-detected project root
    datasets_dir: Path          # datasets/
    samples_dir: Path           # samples/
    runs_dir: Path             # runs/
    examples_dir: Path         # examples/
    yolo_data_yaml: Path       # datasets/ultralytics/data.yaml
```

#### `ModelWeightsConfig`
Stores paths to trained model weights.

```python
class ModelWeightsConfig(BaseModel):
    yolo_weights: Path         # samples/best.pt
    faster_rcnn_weights: Path  # samples/fasterrcnn_best.pt
    detr_weights: Path        # samples/detr_best.pt
```

#### `TrainingConfig`
Training hyperparameters with validation.

```python
class TrainingConfig(BaseModel):
    epochs: int = 20                    # Training epochs
    batch_size: int = 8                 # Batch size
    learning_rate: float = 0.005        # Learning rate
    weight_decay: float = 0.001         # L2 regularization
    image_size: int = 640               # Input image size
    num_workers: int = 4                # DataLoader workers

    @field_validator("epochs")
    def validate_epochs(cls, v):
        if v < 1 or v > 1000:
            raise ValueError("Epochs must be between 1 and 1000")
        return v
```

#### `ScraperConfig`
Web scraper settings.

```python
class ScraperConfig(BaseModel):
    headless: bool = True               # Headless browser mode
    timeout: int = 30000                # Request timeout (ms)
    download_delay: int = 1000          # Delay between requests (ms)
```

#### `Settings` (Root Configuration)
Main settings class that combines all config sections.

```python
class Settings(BaseSettings):
    log_level: str = "INFO"
    paths: PathsConfig
    models: ModelWeightsConfig
    training: TrainingConfig
    scraper: ScraperConfig

    model_config = SettingsConfigDict(
        env_prefix="CCTV_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8"
    )
```

## Environment Variables

### Naming Convention

Environment variables use the `CCTV_` prefix and `__` delimiter for nested settings:

```
CCTV_<SECTION>__<FIELD>
```

### Available Environment Variables

#### General Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CCTV_LOG_LEVEL` | str | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

#### Model Weights

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CCTV_MODELS__YOLO_WEIGHTS` | Path | `samples/best.pt` | YOLOv8 model weights |
| `CCTV_MODELS__FASTER_RCNN_WEIGHTS` | Path | `samples/fasterrcnn_best.pt` | Faster R-CNN weights |
| `CCTV_MODELS__DETR_WEIGHTS` | Path | `samples/detr_best.pt` | DETR weights |

#### Training Parameters

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CCTV_TRAINING__EPOCHS` | int | `20` | Number of training epochs |
| `CCTV_TRAINING__BATCH_SIZE` | int | `8` | Batch size for training |
| `CCTV_TRAINING__LEARNING_RATE` | float | `0.005` | Learning rate |
| `CCTV_TRAINING__WEIGHT_DECAY` | float | `0.001` | Weight decay (L2 regularization) |
| `CCTV_TRAINING__IMAGE_SIZE` | int | `640` | Input image size (pixels) |
| `CCTV_TRAINING__NUM_WORKERS` | int | `4` | DataLoader worker processes |

#### Scraper Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CCTV_SCRAPER__HEADLESS` | bool | `true` | Run browser in headless mode |
| `CCTV_SCRAPER__TIMEOUT` | int | `30000` | Request timeout (milliseconds) |
| `CCTV_SCRAPER__DOWNLOAD_DELAY` | int | `1000` | Delay between requests (milliseconds) |

## Configuration Methods

### Method 1: Using `.env` File (Recommended)

Create a `.env` file in the project root:

```bash
# .env

# General settings
CCTV_LOG_LEVEL=DEBUG

# Model weights (custom paths)
CCTV_MODELS__YOLO_WEIGHTS=/custom/path/yolo_model.pt
CCTV_MODELS__FASTER_RCNN_WEIGHTS=/custom/path/fasterrcnn_model.pt
CCTV_MODELS__DETR_WEIGHTS=/custom/path/detr_model.pt

# Training configuration
CCTV_TRAINING__EPOCHS=50
CCTV_TRAINING__BATCH_SIZE=16
CCTV_TRAINING__LEARNING_RATE=0.001
CCTV_TRAINING__WEIGHT_DECAY=0.0005

# Scraper settings
CCTV_SCRAPER__HEADLESS=false
CCTV_SCRAPER__TIMEOUT=60000
```

The `.env` file is automatically loaded when the application starts.

### Method 2: Export Environment Variables

Set environment variables in your shell:

```bash
# Set individual variables
export CCTV_TRAINING__EPOCHS=30
export CCTV_TRAINING__BATCH_SIZE=8

# Run training
uv run cctv-train
```

### Method 3: Inline Environment Variables

Set variables for a single command:

```bash
# Train with custom epochs
CCTV_TRAINING__EPOCHS=100 uv run cctv-train-detr

# Run UI with debug logging
CCTV_LOG_LEVEL=DEBUG uv run cctv-ui
```

### Method 4: Programmatic Configuration

Modify settings in Python code:

```python
from src.config import get_settings

settings = get_settings()
settings.training.epochs = 50
settings.training.batch_size = 16
```

## Common Configuration Scenarios

### Scenario 1: Training with Custom Dataset

```bash
# .env
CCTV_TRAINING__EPOCHS=30
CCTV_TRAINING__BATCH_SIZE=16
CCTV_TRAINING__LEARNING_RATE=0.001
```

### Scenario 2: Using Pre-trained Weights from Custom Location

```bash
# .env
CCTV_MODELS__YOLO_WEIGHTS=/mnt/models/yolo_v8_custom.pt
CCTV_MODELS__FASTER_RCNN_WEIGHTS=/mnt/models/fasterrcnn_custom.pt
CCTV_MODELS__DETR_WEIGHTS=/mnt/models/detr_custom.pt
```

### Scenario 3: Development/Debug Mode

```bash
# .env
CCTV_LOG_LEVEL=DEBUG
CCTV_TRAINING__EPOCHS=5
CCTV_TRAINING__BATCH_SIZE=2
CCTV_SCRAPER__HEADLESS=false
```

### Scenario 4: High-Performance Training

```bash
# .env
CCTV_TRAINING__EPOCHS=100
CCTV_TRAINING__BATCH_SIZE=32
CCTV_TRAINING__NUM_WORKERS=8
CCTV_TRAINING__LEARNING_RATE=0.01
```

### Scenario 5: Long-Running DETR Training

```bash
# .env
# DETR typically needs 100-300 epochs to converge
CCTV_TRAINING__EPOCHS=150
CCTV_TRAINING__BATCH_SIZE=8
CCTV_TRAINING__LEARNING_RATE=0.0001
```

## Path Configuration

### Default Paths

All paths are automatically resolved relative to the project root:

```text
project_root/
├── datasets/              # Training datasets
│   └── ultralytics/
│       └── data.yaml      # YOLO dataset config
├── samples/               # Model weights
│   ├── best.pt
│   ├── fasterrcnn_best.pt
│   └── detr_best.pt
├── runs/                  # Training results
│   ├── yolo/
│   ├── faster_rcnn/
│   └── detr/
└── examples/              # UI example images
```

### Custom Paths

Override paths using environment variables:

```bash
# Use absolute paths
CCTV_MODELS__YOLO_WEIGHTS=/path/to/custom/yolo.pt

# Or relative to project root
CCTV_MODELS__YOLO_WEIGHTS=custom_models/yolo_v8.pt
```

## Validation

### Type Validation

Pydantic automatically validates types:

```python
# This will raise a ValidationError
CCTV_TRAINING__EPOCHS=invalid  # Must be an integer
```

### Range Validation

Custom validators ensure values are in valid ranges:

```python
# Epochs must be between 1 and 1000
CCTV_TRAINING__EPOCHS=2000  # ValidationError

# Batch size must be positive
CCTV_TRAINING__BATCH_SIZE=-1  # ValidationError
```

### Path Validation

Paths are validated to ensure they exist (for model weights) or are created (for output directories).

## Accessing Configuration

### In Scripts

```python
from src.config import get_settings

settings = get_settings()

# Access nested settings
epochs = settings.training.epochs
model_path = settings.models.yolo_weights

# Access paths
dataset_dir = settings.paths.datasets_dir
```

### In Application Code

Settings are injected via dependency injection:

```python
class TrainingService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def train(self):
        epochs = self.settings.training.epochs
        batch_size = self.settings.training.batch_size
```

## Model-Specific Configuration

### YOLOv8 Configuration

YOLOv8 uses `data.yaml` for dataset configuration:

```yaml
# datasets/ultralytics/data.yaml
path: /absolute/path/to/datasets/ultralytics
train: images/train
val: images/val

nc: 2  # Number of classes
names: ['CCTV', 'CCTV-SIGNS']
```

Training hyperparameters are in `src/config.py`.

### Faster R-CNN Configuration

Faster R-CNN uses DataLoaders with YOLO-format annotations. Configuration is entirely in `src/config.py` and environment variables.

### DETR Configuration

DETR uses COCO-format annotations and HuggingFace Transformers. Configuration is in `src/config.py`.

## Troubleshooting

### Issue: Settings not loading from `.env`

**Solution**: Ensure `.env` is in the project root (same directory as `app.py`).

### Issue: Path not found errors

**Solution**: Use absolute paths or verify relative paths are correct:

```bash
# Use absolute path
CCTV_MODELS__YOLO_WEIGHTS=/full/path/to/model.pt

# Or verify relative path from project root
CCTV_MODELS__YOLO_WEIGHTS=samples/best.pt
```

### Issue: Validation errors on startup

**Solution**: Check environment variable types and ranges:

```bash
# Wrong (string instead of int)
CCTV_TRAINING__EPOCHS=fifty

# Correct
CCTV_TRAINING__EPOCHS=50
```

### Issue: Changes not taking effect

**Solution**: Restart the application after modifying `.env` or environment variables.

## Best Practices

1. **Use `.env` for development**: Keep sensitive or environment-specific config in `.env` (add to `.gitignore`)
2. **Use environment variables for production**: Set variables in deployment environment
3. **Document custom settings**: Comment your `.env` file for team members
4. **Validate early**: Let Pydantic catch configuration errors at startup
5. **Use type hints**: Enable IDE autocomplete and type checking

## Further Reading

- [Architecture Guide](ARCHITECTURE.md) - How configuration integrates with the system
- [Training Guide](TRAINING.md) - Training-specific configuration details
- [Development Guide](DEVELOPMENT.md) - Development environment setup