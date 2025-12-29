# This is the cctv detection application for the UNDO.

## Setup

To run this project you need to create a virtual environment. Open up a terminal and type the following:

```commandline
python3.11 -m venv venv
```

Activate the newly created env:

```commandline
source venv/bin/activate
```

If needed you can deactivate the virtual environment from within root of project:

```commandline
deactivate
```

Install the dependencies for this project:

```commandline
pip install -r requirements.txt
```

## Configuration

The project uses centralized configuration in `src/config.py` with absolute paths resolved from the project root. This allows the application to run from any directory without path issues.

### Default Paths

All paths are automatically resolved relative to the project root:
- **Model weights**: `samples/best.pt`
- **Datasets**: `datasets/`
- **Training results**: `runs/`

### Environment Variables

You can override default paths using environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `CCTV_MODEL_WEIGHTS` | Custom model weights location | `/path/to/custom/model.pt` |
| `CCTV_DATASET_DIR` | Custom dataset directory | `/path/to/custom/dataset` |

### Using .env File

Create a `.env` file in the project root to customize paths (optional):

```bash
# .env
# Override model weights location
CCTV_MODEL_WEIGHTS=/path/to/custom/model.pt

# Override dataset location
CCTV_DATASET_DIR=/path/to/custom/dataset
```

The application will automatically load these settings if the file exists.

## Image labelling
To label images you need to have [Docker](https://www.docker.com) and [label-studio](https://labelstud.io) installed.

From a terminal run the following to pull the latest label-studio image:

```commandline
docker pull heartexlabs/label-studio:latest
```

To run the container:

```commandline
docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest
```

Open your web browser and navigate to http://localhost:8080.
Upon first access, you’ll be prompted to create a username and password. 
This account is for local use only and does not require any external registration.

## Model training results

### Overview
We trained a YOLO model on our custom dataset for 20 epochs. The final evaluation metrics show strong performance with high precision, recall, and mAP, particularly at an IoU threshold of 0.5.

| Metric             | Value    |
|--------------------|----------|
| Epochs             | 20       |
| Training Loss      | 4.99     |
| Validation Loss    | 5.66     |
| Precision (B)      | 0.841    |
| Recall (B)         | 0.838    |
| mAP@0.5 (B)        | 0.873    |
| mAP@0.5:0.95 (B)   | 0.415    |

### Loss curves
Training and validation losses decreased steadily across epochs, showing no major overfitting:

- **Training Loss:** Decreased from ~8.0 to ~5.0.
- **Validation Loss:** Decreased from ~6.9 to ~5.7.
- **Gap between train and validation loss remained stable**, suggesting good generalization.

### Detection metrics
- **Precision** and **recall** improved steadily, reaching balanced high scores (above 83%).
- **mAP@0.5** (object detection quality at 0.5 IoU threshold) reached **87%**.
- **mAP@0.5–0.95** (stricter localization accuracy) reached **41%**, indicating moderate room for improvement in bounding box precision.

### Visualizations
Results
![Results](samples/results.png)

Training batch images
![Train batch images sample 1](samples/train_batch2.jpg)
![Train batch images sample 2](samples/train_batch351.jpg)

Valuation prediction samples
![Valuation batch sample 1](samples/val_batch0_pred.jpg)
![Valuation batch sample 2](samples/val_batch1_pred.jpg)

---

### Disclaimer
Although most images used in training have been collected through ethnographic research,
some images used in this project come from the dataset of the [Fuziih CCTV-Exposure](https://github.com/Fuziih/cctv-exposure/tree/main)

> Contact us if you want access to the dataset or the weights of the model.

## User Interface

The application has a User Interface for uploading images and using the custom trained YOLOv8 to detect CCTV images.

### Running the UI

Thanks to the centralized configuration system, you can run the application from any directory:

```commandline
# From project root
python app.py

# OR from src/presentation directory
cd src/presentation && python main_ui.py

# OR from anywhere in the project
python src/presentation/main_ui.py
```

Then open up a browser and visit: `http://127.0.0.1:7860`.

## Testing

This project uses pytest for comprehensive testing with >60% code coverage.

### Running Tests

Run all tests:
```commandline
pytest tests/
```

Run tests with coverage report:
```commandline
pytest tests/ --cov=src --cov-report=term-missing
```

Run tests with HTML coverage report:
```commandline
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in your browser
```

Run specific test file:
```commandline
pytest tests/domain/services/test_distance_calculator.py
```

Run tests from local script:
```commandline
bash local_test_pipeline.sh
```

### Test Structure

The test suite mirrors the source code structure:
```
tests/
├── domain/           # Domain layer tests (100% coverage)
├── application/      # Application layer tests (100% coverage)
├── infrastructure/   # Infrastructure layer tests (90%+ coverage)
└── presentation/     # Presentation layer tests
```

### Test Coverage

Current coverage: **65.13%** (120 tests)
- Domain services: 100% coverage
- Application services: 100% coverage
- Infrastructure layer: 90%+ coverage

## Code Quality

This project uses automated tools to ensure code quality and consistency.

### Pre-commit Hooks (Required)

Install pre-commit hooks to automatically check code before committing:

```commandline
pre-commit install
```

The hooks will run:
- **Ruff linter** - Fast Python linting and code quality checks
- **Ruff formatter** - Automatic code formatting

### Optional: MyPy Type Checking

MyPy is available for optional local type checking but is NOT part of pre-commit hooks:

```commandline
# Run type checking manually (requires full virtual environment)
mypy src/
```

**Note**: We rely on comprehensive test coverage (72%) and Ruff for code quality in the commit workflow. Type hints in the code still provide IDE support and documentation value.