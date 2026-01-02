# 🏗️ Architecture Guide

This document provides a detailed overview of the CCTV Detection System's architecture, design patterns, and implementation details.

## Overview

The project follows a **clean layered architecture** with clear separation of concerns, ensuring maintainability, testability, and extensibility.

## Architectural Layers

### 1. Domain Layer (`src/domain/`)

The innermost layer containing core business logic and abstract interfaces (contracts). This layer has no dependencies on external frameworks or libraries.

#### Services (Abstract Interfaces)

**Model Training & Detection**
- `ModelTrainer`: Interface for training detection models
  - Methods: `train(data_path, epochs, batch_size, ...)`
  - Implementations: YOLOv8, Faster R-CNN, DETR trainers

- `ObjectDetector`: Interface for running object detection inference
  - Methods: `detect(image_path, confidence_threshold)`
  - Implementations: YOLOv8, Faster R-CNN, DETR detectors

**Data Processing**
- `DatasetSplitter`: Interface for splitting datasets into train/val/test sets
  - Methods: `split(source_dir, train_ratio, val_ratio, test_ratio)`

- `ImageConverter`: Interface for image format conversion
  - Methods: `convert(input_path, output_format)`
  - Use case: Converting HEIF images to JPEG

- `DatasetPreparer`: Interface for preparing datasets for training
  - Methods: `prepare(source_dir, output_dir, format)`
  - Handles format conversions (YOLO → COCO, etc.)

**Surveillance Calculations**
- `DistanceCalculator`: Implements DORI standard calculations
  - Based on IEC 62676-4 for surveillance camera planning
  - Calculates Detection, Observation, Recognition, Identification distances
  - Methods: `calculate_dori_distances(sensor_specs, lens_specs)`

#### Entities

- `Camera`: Domain model representing a surveillance camera
  - Attributes: position, orientation, field_of_view, specifications
  - Methods: domain-specific behaviors

#### Exceptions

Custom exception hierarchy for domain-specific error handling:
- `CCTVDetectionError`: Base exception
- `ModelNotFoundError`: Model weights not found
- `InvalidDatasetError`: Dataset validation failures
- `InferenceError`: Detection inference failures

### 2. Infrastructure Layer (`src/infrastructure/`)

Concrete implementations of domain interfaces using specific frameworks and libraries.

#### Model Trainers

- **`YoloUltralyticsTrainer`**: YOLOv8 training using Ultralytics library
  - Hyperparameters: image_size=640, lr=0.005, weight_decay=0.001
  - Uses mosaic (0.8) and mixup (0.1) augmentation
  - Freezes layers [0, 1, 2, 3] for transfer learning

- **`FasterRCNNTrainer`**: Faster R-CNN using PyTorch/torchvision
  - Based on ResNet-50 FPN backbone
  - Handles YOLO-format annotations via custom DataLoaders
  - Automatic mAP computation using COCO evaluation protocol

- **`DETRTrainer`**: DETR transformer using Hugging Face Transformers
  - Pre-trained facebook/detr-resnet-50 model
  - Fine-tuned on COCO-format annotations
  - Automatic mAP computation post-training

#### Object Detectors

- **`YoloDetector`**: YOLOv8 inference wrapper
- **`FasterRCNNDetector`**: Faster R-CNN inference wrapper
- **`DETRDetector`**: DETR inference wrapper

Each detector implements:
- `detect(image_path, confidence_threshold)` → List[Detection]
- Device selection (MPS, CUDA, CPU)
- Bounding box format normalization

#### Factory Pattern

- **`DetectorFactory`**: Creates detector instances based on model type
  ```python
  factory = DetectorFactory()
  detector = factory.create_detector("yolo", model_path="samples/best.pt")
  ```

#### Data Processing

- **`RandomSplitter`**: Random dataset splitting implementation
- **`ImageConverterImpl`**: HEIF → JPEG conversion using pillow-heif
- **`DatasetPreparerImpl`**: Dataset format conversion orchestration

#### Data Loaders

Custom PyTorch DataLoaders for each model:
- `CCTVDataset`: Loads YOLO-format annotations
- `CocoDataset`: Loads COCO-format annotations
- Custom collate functions for batching

#### Web Scraping

- **`ImageScraper`**: Camera image downloading using Playwright
  - Headless browser automation
  - Handles dynamic content and JavaScript-heavy pages

### 3. Application Layer (`src/application/`)

Orchestrates domain and infrastructure components to implement use cases.

#### Services

- **`TrainingService`**: Orchestrates model training workflow
  - Validates dataset
  - Creates data loaders
  - Manages device selection (MPS → CUDA → CPU)
  - Monitors training progress

- **`SurveillanceService`**: Provides surveillance-related calculations
  - Uses `DistanceCalculator` for DORI computations
  - Camera placement optimization

- **`DatasetPreparation`**: Dataset preparation workflow
  - Coordinates splitting, conversion, validation

- **`CameraImageDownloader`**: Camera image downloading workflow
  - Uses `ImageScraper` to collect training data
  - Handles rate limiting and error recovery

### 4. UI Layer (`src/ui/`)

User interfaces for interacting with the system.

#### Gradio Web Application (`gradio_app.py`)

Multi-tab interface with:

1. **Single Model Detection**: Upload image, select model, adjust confidence
2. **Model Comparison**: Side-by-side results from all three models
3. **Example Images**: Pre-loaded gallery for quick testing
4. **Performance Dashboard**: Training metrics and visualizations
5. **About**: Model details and citation info

Key functions:
- `create_demo()`: Builds Gradio interface
- `launch_ui()`: Starts web server

### 5. Scripts Layer (`scripts/`)

Command-line utilities accessible via console scripts.

#### Training Scripts
- `train_yolo.py`: YOLOv8 training orchestration
- `train_faster_rcnn.py`: Faster R-CNN training orchestration
- `train_detr.py`: DETR training orchestration

#### Evaluation Scripts
- `evaluate_faster_rcnn.py`: Compute mAP for trained Faster R-CNN
- `evaluate_detr.py`: Compute mAP for trained DETR

#### Utility Scripts
- `prepare_examples.py`: Copy validation images to examples/
- `benchmark.py`: Benchmark inference speed across models

## Design Patterns

### Dependency Injection

Services receive dependencies through constructors, enabling:
- Easy testing with mocks
- Flexible configuration
- Clear dependency graphs

```python
# Application layer
class TrainingService:
    def __init__(self, trainer: ModelTrainer, splitter: DatasetSplitter):
        self.trainer = trainer
        self.splitter = splitter
```

### Strategy Pattern

Multiple implementations of the same interface allow runtime selection:

```python
# Select trainer based on model type
if model_type == "yolo":
    trainer = YoloUltralyticsTrainer()
elif model_type == "faster_rcnn":
    trainer = FasterRCNNTrainer()
elif model_type == "detr":
    trainer = DETRTrainer()

# All implement the same ModelTrainer interface
trainer.train(data_path, epochs=20, batch_size=8)
```

### Factory Pattern

`DetectorFactory` encapsulates object creation logic:

```python
factory = DetectorFactory()
yolo_detector = factory.create_detector("yolo")
fasterrcnn_detector = factory.create_detector("faster_rcnn")
detr_detector = factory.create_detector("detr")
```

### Repository Pattern (Implicit)

Data loaders act as repositories for training data:
- Abstract data access
- Handle format conversions
- Provide consistent interfaces

## Configuration Management

### Type-Safe Configuration (Pydantic v2)

All configuration in `src/config.py`:

```python
class Settings(BaseSettings):
    paths: PathsConfig
    models: ModelWeightsConfig
    training: TrainingConfig
    scraper: ScraperConfig

    model_config = SettingsConfigDict(
        env_prefix="CCTV_",
        env_nested_delimiter="__"
    )
```

Benefits:
- Runtime validation
- Type checking
- Environment variable support
- Default values with overrides

## Device Selection Strategy

Automatic device selection in priority order:

1. **Apple Metal Performance Shaders (MPS)**: M1/M2/M3 Macs
2. **CUDA**: NVIDIA GPUs
3. **CPU**: Fallback for all systems

```python
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

## Data Flow

### Training Flow

```
User → train_yolo.py
  ↓
TrainingService.train()
  ↓
1. DatasetSplitter.split()        # Split dataset
2. YoloUltralyticsTrainer.train() # Train model
3. Save weights to samples/
  ↓
Training metrics saved to runs/
```

### Inference Flow

```
User uploads image → Gradio UI
  ↓
DetectorFactory.create_detector(model_type)
  ↓
Detector.detect(image, confidence)
  ↓
Results displayed with bounding boxes
```

## Project Structure

```
cctv_detection/
├── src/
│   ├── domain/                    # Core business logic
│   │   ├── services/             # Abstract interfaces
│   │   │   ├── model_trainer.py
│   │   │   ├── object_detector.py
│   │   │   ├── data_splitter.py
│   │   │   ├── image_converter.py
│   │   │   ├── dataset_preparer.py
│   │   │   └── distance_calculator.py
│   │   ├── camera.py             # Domain entities
│   │   └── exceptions.py         # Custom exceptions
│   │
│   ├── infrastructure/           # Concrete implementations
│   │   ├── yolo_trainer.py
│   │   ├── faster_rcnn_trainer.py
│   │   ├── detr_trainer.py
│   │   ├── yolo_detector.py
│   │   ├── faster_rcnn_detector.py
│   │   ├── detr_detector.py
│   │   ├── detector_factory.py
│   │   ├── data_loaders.py
│   │   ├── splitters.py
│   │   ├── image_converter_impl.py
│   │   ├── dataset_preparer_impl.py
│   │   └── image_scraper.py
│   │
│   ├── application/              # Use case orchestration
│   │   ├── training_service.py
│   │   ├── surveillance_service.py
│   │   ├── dataset_preparation.py
│   │   └── camera_image_downloader.py
│   │
│   ├── ui/                       # User interfaces
│   │   └── gradio_app.py
│   │
│   └── config.py                 # Configuration
│
├── scripts/                      # CLI utilities
├── tests/                        # Test suite
└── app.py                        # Entry point
```

## Testing Strategy

- **Domain Layer**: 100% coverage (pure business logic)
- **Application Layer**: 100% coverage (orchestration)
- **Infrastructure Layer**: 90%+ coverage (integration tests)
- **UI Layer**: Component testing with mocks

See [Development Guide](DEVELOPMENT.md) for testing details.

## Extension Points

### Adding a New Model

1. Create trainer in `infrastructure/`:
   ```python
   class NewModelTrainer(ModelTrainer):
       def train(self, ...): ...
   ```

2. Create detector in `infrastructure/`:
   ```python
   class NewModelDetector(ObjectDetector):
       def detect(self, ...): ...
   ```

3. Update `DetectorFactory`
4. Add training script in `scripts/`
5. Update UI in `gradio_app.py`

### Adding a New Data Format

1. Implement `DatasetPreparer` for the format
2. Create custom DataLoader if needed
3. Update conversion utilities

## Best Practices

1. **Keep domain layer framework-agnostic**: No PyTorch, TensorFlow, or framework-specific code
2. **Use dependency injection**: Pass dependencies explicitly
3. **Type hints everywhere**: Enable IDE support and catch errors early
4. **Test at boundaries**: Integration tests for infrastructure, unit tests for domain
5. **Configuration as code**: Use Pydantic settings, not config files

## Further Reading

- [Configuration Guide](CONFIGURATION.md) - Detailed configuration options
- [Training Guide](TRAINING.md) - Training workflow and hyperparameters
- [Development Guide](DEVELOPMENT.md) - Contributing and testing