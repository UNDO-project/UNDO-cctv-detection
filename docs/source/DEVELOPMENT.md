# 💻 Development Guide

This guide covers setting up your development environment, running tests, maintaining code quality, and contributing to the CCTV Detection System.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- uv package manager

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UNDO-project/UNDO-cctv-detection.git
   cd UNDO-cctv-detection
   ```

2. **Install dependencies**:
   ```bash
   # Install all dependencies including dev tools
   uv sync --all-extras
   ```

3. **Install pre-commit hooks**:
   ```bash
   # Install hooks for automatic code formatting
   uv run pre-commit install
   ```

4. **Verify installation**:
   ```bash
   # Run tests to ensure everything works
   uv run pytest tests/
   ```

## Project Structure

```
cctv_detection/
├── src/
│   ├── domain/              # Core business logic
│   ├── application/         # Use case orchestration
│   ├── infrastructure/      # Concrete implementations
│   ├── ui/                  # Gradio web interface
│   └── config.py            # Configuration
├── scripts/                 # CLI utilities
├── tests/                   # Test suite
│   ├── domain/             # Domain layer tests
│   ├── application/        # Application layer tests
│   ├── infrastructure/     # Infrastructure tests
│   └── ui/                 # UI tests
├── docs/                    # Documentation
├── datasets/                # Training data
├── samples/                 # Model weights
└── examples/                # UI example images
```

## Testing

### Running Tests

The project uses pytest for comprehensive testing with >65% code coverage.

#### Run All Tests

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run with output (show print statements)
uv run pytest tests/ -s
```

#### Run Specific Tests

```bash
# Run specific test file
uv run pytest tests/infrastructure/test_detectors.py

# Run specific test function
uv run pytest tests/domain/test_services.py::test_model_trainer

# Run tests matching pattern
uv run pytest tests/ -k "detector"
```

#### Coverage Reports

```bash
# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
uv run pytest tests/ --cov=src --cov-report=html

# Open coverage report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

#### Test Pipeline Script

```bash
# Run comprehensive test pipeline
bash local_test_pipeline.sh
```

### Test Structure

Tests mirror the source code structure:

```
tests/
├── domain/
│   ├── test_camera.py           # Domain entity tests
│   └── test_services.py         # Service interface tests
├── application/
│   ├── test_training_service.py
│   └── test_surveillance_service.py
├── infrastructure/
│   ├── test_trainers.py         # Trainer implementation tests
│   ├── test_detectors.py        # Detector implementation tests
│   └── test_data_loaders.py     # DataLoader tests
└── ui/
    └── test_gradio_app.py       # UI component tests
```

### Test Coverage Goals

| Layer | Coverage Goal | Current |
|-------|---------------|---------|
| Domain | 100% | 100% |
| Application | 100% | 100% |
| Infrastructure | 90%+ | 90%+ |
| UI | 60%+ | 60%+ |
| **Overall** | **65%+** | **65%+** |

### Writing Tests

#### Test Template

```python
import pytest
from src.domain.services.model_trainer import ModelTrainer

class TestModelTrainer:
    """Test suite for ModelTrainer interface."""

    def test_train_with_valid_params(self):
        """Test training with valid parameters."""
        trainer = MockModelTrainer()
        result = trainer.train(
            data_path="datasets/",
            epochs=10,
            batch_size=8
        )
        assert result is not None

    def test_train_with_invalid_epochs(self):
        """Test training fails with invalid epochs."""
        trainer = MockModelTrainer()
        with pytest.raises(ValueError):
            trainer.train(data_path="datasets/", epochs=-1)
```

#### Using Fixtures

```python
@pytest.fixture
def sample_detector():
    """Create a sample detector for testing."""
    return YoloDetector(model_path="samples/best.pt")

def test_detection(sample_detector):
    """Test object detection."""
    results = sample_detector.detect("test_image.jpg")
    assert len(results) > 0
```

#### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

def test_training_with_mock():
    """Test training service with mocked trainer."""
    mock_trainer = Mock(spec=ModelTrainer)
    mock_trainer.train.return_value = {"loss": 0.5, "mAP": 0.85}

    service = TrainingService(trainer=mock_trainer)
    result = service.train_model()

    mock_trainer.train.assert_called_once()
    assert result["mAP"] == 0.85
```

## Code Quality

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit to ensure code quality.

#### Configured Hooks

- **Ruff linter**: Fast Python linting and code quality checks
- **Ruff formatter**: Automatic code formatting

#### Manual Execution

```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files

# Skip hooks for a commit (not recommended)
git commit --no-verify -m "Message"
```

### Linting

#### Run Ruff Linter

```bash
# Check all files
uv run ruff check src/

# Check specific file
uv run ruff check src/domain/camera.py

# Auto-fix issues
uv run ruff check src/ --fix
```

### Formatting

#### Run Ruff Formatter

```bash
# Format all files
uv run ruff format src/

# Format specific file
uv run ruff format src/infrastructure/yolo_detector.py

# Check formatting without modifying
uv run ruff format src/ --check
```

### Type Checking (Optional)

MyPy is available for optional type checking:

```bash
# Run type checking
uv run mypy src/

# Check specific file
uv run mypy src/domain/services/model_trainer.py

# Strict mode
uv run mypy src/ --strict
```

**Note**: Type checking is optional. We rely on comprehensive test coverage and Ruff for code quality in the commit workflow.

## Coding Standards

### Docstring Format

This project uses **reStructuredText (reST)** format for all docstrings:

```python
def train_model(data_path: str, epochs: int, batch_size: int = 8) -> dict:
    """Train an object detection model.

    This function trains a model on the provided dataset with specified
    hyperparameters and returns training metrics.

    :param data_path: Path to training dataset
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training (default: 8)
    :return: Dictionary containing training metrics (loss, mAP, etc.)
    :rtype: dict
    :raises ValueError: If epochs is less than 1
    :raises FileNotFoundError: If data_path does not exist
    """
    pass
```

**Do NOT use**:
- Google-style (`Args:`, `Returns:`)
- NumPy-style docstrings

### Type Hints

Use type hints for all function signatures:

```python
from pathlib import Path
from typing import List, Optional, Dict

def detect_objects(
    image_path: Path,
    confidence: float = 0.25,
    classes: Optional[List[str]] = None
) -> Dict[str, any]:
    """Detect objects in an image."""
    pass
```

### Naming Conventions

- **Classes**: PascalCase (`YoloDetector`, `TrainingService`)
- **Functions/Methods**: snake_case (`train_model`, `get_device`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_CONFIDENCE`, `MAX_EPOCHS`)
- **Private**: Prefix with underscore (`_internal_method`)

### Import Organization

```python
# Standard library
import os
from pathlib import Path
from typing import List

# Third-party
import torch
import numpy as np
from ultralytics import YOLO

# Local
from src.domain.services.model_trainer import ModelTrainer
from src.config import get_settings
```

### Architecture Principles

1. **Dependency Inversion**: Domain layer has no dependencies on infrastructure
2. **Single Responsibility**: Each class has one clear purpose
3. **Open/Closed**: Open for extension, closed for modification
4. **Interface Segregation**: Small, focused interfaces
5. **Dependency Injection**: Pass dependencies explicitly

See [Architecture Guide](ARCHITECTURE.md) for details.

## Git Workflow

### Branching Strategy

```
main                     # Stable, production-ready code
  ├── develop            # Integration branch
  │   ├── feature/xyz    # New features
  │   ├── bugfix/abc     # Bug fixes
  │   └── refactor/def   # Code refactoring
```

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:

```
feat(detector): add DETR detector implementation

Implement DETRDetector class using HuggingFace Transformers.
Includes preprocessing, inference, and postprocessing logic.

Closes #42
```

```
fix(training): resolve CUDA out of memory error

Reduce default batch size from 16 to 8 to prevent OOM
on GPUs with 8GB memory.
```

### Pull Request Process

1. **Create feature branch**:
   ```bash
   git checkout -b feature/new-detector
   ```

2. **Make changes and commit**:
   ```bash
   git add .
   git commit -m "feat(detector): add new detector"
   ```

3. **Push to remote**:
   ```bash
   git push origin feature/new-detector
   ```

4. **Create PR**:
   - Go to GitHub repository
   - Click "New Pull Request"
   - Fill in template with:
     - Description of changes
     - Related issues
     - Testing done
     - Screenshots (if UI changes)

5. **Address review comments**:
   ```bash
   # Make requested changes
   git add .
   git commit -m "refactor: address review comments"
   git push origin feature/new-detector
   ```

6. **Merge**: After approval, squash and merge

## Adding New Features

### Adding a New Model

1. **Create trainer** in `src/infrastructure/`:
   ```python
   class NewModelTrainer(ModelTrainer):
       """Trainer for NewModel architecture."""

       def train(self, data_path: str, epochs: int, ...) -> dict:
           """Train NewModel."""
           pass
   ```

2. **Create detector** in `src/infrastructure/`:
   ```python
   class NewModelDetector(ObjectDetector):
       """Detector for NewModel architecture."""

       def detect(self, image_path: str, confidence: float) -> List[Detection]:
           """Run detection."""
           pass
   ```

3. **Update factory** in `detector_factory.py`:
   ```python
   def create_detector(self, model_type: str) -> ObjectDetector:
       if model_type == "newmodel":
           return NewModelDetector(model_path=...)
   ```

4. **Add training script** in `scripts/`:
   ```python
   # scripts/train_newmodel.py
   def main():
       trainer = NewModelTrainer()
       trainer.train(...)
   ```

5. **Update UI** in `src/ui/gradio_app.py`

6. **Add tests** in `tests/infrastructure/`:
   ```python
   class TestNewModelDetector:
       def test_detect(self):
           detector = NewModelDetector()
           results = detector.detect("test.jpg")
           assert len(results) > 0
   ```

7. **Update documentation**:
   - `docs/TRAINING.md`
   - `docs/ARCHITECTURE.md`
   - `README.md`

### Adding a New Data Format

1. **Implement converter** in `src/infrastructure/`
2. **Update data loaders**
3. **Add tests**
4. **Update `docs/DATASET.md`**

## Debugging

### Debug Mode

Enable debug logging:

```bash
CCTV_LOG_LEVEL=DEBUG uv run python app.py
```

### Common Issues

#### Import Errors

**Solution**: Ensure you're in the project root and virtual env is activated

```bash
cd /path/to/cctv_detection
source .venv/bin/activate  # Or uv handles this automatically
```

#### Test Failures

**Solution**: Check test output, run with `-v` for details

```bash
uv run pytest tests/ -v
```

#### Type Errors

**Solution**: Run mypy to find type issues

```bash
uv run mypy src/
```

### VSCode Configuration

Recommended `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "none",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"]
}
```

## Performance Profiling

### Profile Training

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Training code here
trainer.train(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)  # Top 20 functions
```

### Profile Inference

```python
import time

start = time.time()
results = detector.detect("image.jpg")
elapsed = time.time() - start

print(f"Inference time: {elapsed*1000:.2f}ms")
```

## Documentation

### Building Sphinx Docs

```bash
cd docs/
make html
open build/html/index.html
```

### Updating Guides

When updating documentation:

1. Edit markdown files in `docs/`
2. Ensure links work
3. Update table of contents if needed
4. Commit with `docs:` prefix

## Contributing

### How to Contribute

1. **Find an issue**: Check GitHub Issues or create one
2. **Discuss approach**: Comment on issue before starting
3. **Follow guidelines**: Use this development guide
4. **Write tests**: Maintain >65% coverage
5. **Update docs**: Document new features
6. **Submit PR**: Follow PR template

### Code Review

PRs must:
- Pass all tests
- Pass linting/formatting checks
- Maintain or improve coverage
- Include documentation updates
- Have clear commit messages

## Release Process

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG**
3. **Run full test suite**
4. **Build package**: `uv build`
5. **Create release tag**: `git tag v1.0.0`
6. **Push to GitHub**: `git push --tags`

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search GitHub Issues
- **Questions**: Open a GitHub Discussion
- **Email**: Contact UNDO project team

## Best Practices Summary

1. **Test everything**: Aim for >65% coverage
2. **Use type hints**: Enable IDE support and catch errors
3. **Write docstrings**: Use reST format
4. **Follow architecture**: Respect layer boundaries
5. **Commit often**: Small, focused commits
6. **Review code**: Self-review before submitting PR
7. **Update docs**: Documentation is code

## Further Reading

- [Architecture Guide](ARCHITECTURE.md) - System design and patterns
- [Configuration Guide](CONFIGURATION.md) - Environment setup
- [Training Guide](TRAINING.md) - Model training
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)