# 📹 CCTV Detection System

![License](https://img.shields.io/badge/license-GPL--3.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

> Part of the [Understanding Nordic Digital Order (UNDO)](https://undo-project.info) project

A multi-model object detection system for identifying CCTV cameras and signage using state-of-the-art deep learning architectures: YOLOv8, Faster R-CNN, and DETR.

## ✨ Features

- 🎯 **Multi-Model Support**: Compare three detection architectures (YOLOv8, Faster R-CNN, DETR)
- 🎨 **Interactive Web UI**: Gradio-based interface with side-by-side model comparison
- 📊 **Performance Dashboard**: Training metrics, loss curves, and mAP comparisons
- 🧪 **Comprehensive Testing**: >65% code coverage with 190+ passing tests
- ⚙️ **Type-Safe Configuration**: Pydantic v2 settings with environment variable support

## 🚀 Quick Start

### Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast package management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

### 📦 Download Model Weights

Place trained model weights in the `samples/` directory:

```
samples/
├── best.pt              # YOLOv8 weights
├── fasterrcnn_best.pt   # Faster R-CNN weights
└── detr_best.pt         # DETR weights
```

Or train models from scratch (see [Training Guide](docs/source/TRAINING.md)).

> **Need access to pre-trained weights or the full dataset?** Contact the UNDO project team.

### 🎬 Launch the Application

```bash
# Start the Gradio web interface
uv run python app.py

# Or use the console script
uv run cctv-ui
```

Visit `http://127.0.0.1:7860` in your browser.

## 📚 Documentation

Comprehensive guides are available in the `docs/source/` directory:

| Guide | Description |
|-------|-------------|
| 🏗️ [Architecture](docs/source/ARCHITECTURE.md) | Layered architecture, design patterns, and project structure |
| ⚙️ [Configuration](docs/source/CONFIGURATION.md) | Environment variables, paths, and settings customization |
| 🧠 [Training](docs/source/TRAINING.md) | Training all three models with custom datasets |
| 📈 [Evaluation](docs/source/EVALUATION.md) | Model evaluation and performance comparison methodology |
| 🎛️ [UI Guide](docs/source/UI_GUIDE.md) | Using the web interface and preparing example images |
| 💻 [Development](docs/source/DEVELOPMENT.md) | Testing, code quality, and contributing guidelines |
| 🗂️ [Dataset](docs/source/DATASET.md) | Dataset structure, format conversion, and labeling |

## 🛠️ Console Scripts

| Command | Description |
|---------|-------------|
| `uv run cctv-ui` | Launch Gradio web interface |
| `uv run cctv-train` | Train YOLOv8 model |
| `uv run cctv-train-faster-rcnn` | Train Faster R-CNN model |
| `uv run cctv-train-detr` | Train DETR model |
| `uv run cctv-evaluate-faster-rcnn` | Evaluate Faster R-CNN and compute mAP |
| `uv run cctv-evaluate-detr` | Evaluate DETR and compute mAP |
| `uv run cctv-prepare-examples` | Prepare example images for UI |
| `uv run cctv-benchmark` | Benchmark inference speed |

## 🗂️ Project Structure

```
cctv_detection/
├── app.py                   # Main entry point
├── src/
│   ├── domain/             # Business logic & interfaces
│   ├── application/        # Use case orchestration
│   ├── infrastructure/     # Concrete implementations
│   ├── ui/                 # Gradio web interface
│   ├── tools/              # Auxiliary tooling (e.g. data collection / scraping)
│   └── config.py           # Centralized configuration
├── scripts/                # Training and utility scripts
├── tests/                  # Comprehensive test suite
├── docs/                   # Detailed documentation
├── datasets/               # Training data
├── samples/                # Model weights
└── examples/               # UI example images
```

See [Architecture Guide](docs/source/ARCHITECTURE.md) for detailed information.

## 🤝 Contributing

Contributions are welcome! Please see the [Development Guide](docs/source/DEVELOPMENT.md) for:
- Setting up your development environment
- Running tests and code quality checks
- Coding standards and best practices
- Submitting pull requests

## 📄 License

This project is licensed under the GNU General Public License v3.0 - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [UNDO Project](https://undo-project.info) team
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Fuziih CCTV-Exposure Dataset](https://github.com/Fuziih/cctv-exposure)

## 📚 Citation

If you use this system in research, please cite:

```bibtex
@software{cctv_detection_2024,
  title={CCTV Detection System},
  author={UNDO Project},
  year={2024},
  url={https://github.com/UNDO-project/UNDO-cctv-detection}
}
```