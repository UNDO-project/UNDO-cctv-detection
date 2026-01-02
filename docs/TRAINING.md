# 🧠 Training Guide

This guide covers training all three object detection models (YOLOv8, Faster R-CNN, DETR) on custom CCTV datasets.

## Overview

The CCTV Detection System supports training three state-of-the-art object detection architectures:
- **YOLOv8**: Fast single-stage CNN detector
- **Faster R-CNN**: Accurate two-stage CNN detector
- **DETR**: Modern transformer-based detector

Each model has different characteristics, convergence speeds, and optimal training configurations.

## Prerequisites

### Dataset Requirements

Ensure your dataset is properly structured:

```
datasets/
├── images/          # Raw images
├── labels/          # YOLO-format labels
├── ultralytics/     # Prepared for YOLO training
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── classes.txt      # Class names (CCTV, CCTV-SIGNS)
└── notes.json       # Dataset metadata
```

See [Dataset Guide](DATASET.md) for dataset preparation details.

### System Requirements

- Python 3.11+
- 8GB+ RAM (16GB+ recommended for DETR)
- GPU support (optional but recommended):
  - Apple Silicon (M1/M2/M3) with MPS
  - NVIDIA GPU with CUDA
  - CPU fallback available

## Training YOLOv8

### Quick Start

```bash
# Train with default settings (20 epochs)
uv run cctv-train

# Or run directly
uv run python scripts/train_yolo.py
```

### Configuration

YOLOv8 training is configured via:
1. `src/config.py` - Default hyperparameters
2. `datasets/ultralytics/data.yaml` - Dataset configuration
3. Environment variables - Runtime overrides

#### Default Hyperparameters

```python
# In src/config.py
TrainingConfig(
    epochs=20,
    batch_size=8,
    learning_rate=0.005,      # Reduced for small datasets
    weight_decay=0.001,       # L2 regularization
    image_size=640,
    num_workers=4
)
```

#### YOLOv8-Specific Settings

The `YoloUltralyticsTrainer` uses these additional settings:

```python
model.train(
    data=data_yaml_path,
    epochs=epochs,
    imgsz=640,
    batch=batch_size,
    lr0=0.005,              # Initial learning rate
    weight_decay=0.001,
    mosaic=0.8,             # Mosaic augmentation probability
    mixup=0.1,              # Mixup augmentation probability
    freeze=[0, 1, 2, 3],    # Freeze backbone layers (transfer learning)
    device='mps'            # or 'cuda' or 'cpu'
)
```

### Custom Training

```bash
# Train for more epochs
CCTV_TRAINING__EPOCHS=50 uv run cctv-train

# Use larger batch size (if GPU memory allows)
CCTV_TRAINING__BATCH_SIZE=16 uv run cctv-train

# Combine multiple settings
CCTV_TRAINING__EPOCHS=30 CCTV_TRAINING__BATCH_SIZE=16 uv run cctv-train
```

### Training Output

Results are saved to `runs/detect/train/`:

```
runs/detect/train/
├── weights/
│   ├── best.pt          # Best model weights
│   └── last.pt          # Last epoch weights
├── results.png          # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── results.csv          # Detailed metrics
```

Copy best weights to `samples/`:

```bash
cp runs/detect/train/weights/best.pt samples/best.pt
```

### YOLOv8 Training Results

Our YOLOv8 model achieved these results after 20 epochs:

| Metric | Value |
|--------|-------|
| Precision (B) | 0.841 |
| Recall (B) | 0.838 |
| mAP@0.5 (B) | 0.873 |
| mAP@0.5:0.95 (B) | 0.415 |

**Loss Curves:**
- Training Loss: 8.0 → 5.0
- Validation Loss: 6.9 → 5.7
- Stable gap indicates good generalization

## Training Faster R-CNN

### Quick Start

```bash
# Train with default settings (20 epochs)
uv run cctv-train-faster-rcnn

# Or run directly
uv run python scripts/train_faster_rcnn.py
```

### Configuration

Faster R-CNN training uses:
- YOLO-format annotations (converted internally)
- PyTorch DataLoaders with custom collate functions
- ResNet-50 FPN backbone (pre-trained on COCO)

#### Default Hyperparameters

```python
# In FasterRCNNTrainer
TrainingConfig(
    epochs=20,
    batch_size=8,
    learning_rate=0.005,
    weight_decay=0.0005,
    momentum=0.9,
    step_size=10,           # LR scheduler step
    gamma=0.1               # LR decay factor
)
```

### Custom Training

```bash
# Train for 50 epochs (recommended for convergence)
CCTV_TRAINING__EPOCHS=50 uv run cctv-train-faster-rcnn

# Adjust learning rate
CCTV_TRAINING__LEARNING_RATE=0.001 uv run cctv-train-faster-rcnn
```

### Training Output

Results are saved to `runs/faster_rcnn/train/`:

```
runs/faster_rcnn/train/
├── model_epoch_*.pth        # Checkpoint every 5 epochs
├── training_metrics.json    # mAP, loss history
├── loss_curves.png          # Training/val loss plots
└── logs/                    # TensorBoard logs
```

Best model is automatically saved to:
```bash
samples/fasterrcnn_best.pt
```

### Evaluation

Faster R-CNN training automatically computes mAP@0.5 and mAP@0.5:0.95 using COCO evaluation protocol.

To re-evaluate an already-trained model:

```bash
uv run cctv-evaluate-faster-rcnn
```

## Training DETR

### Quick Start

```bash
# Train with default settings (20 epochs)
uv run cctv-train-detr

# Or run directly
uv run python scripts/train_detr.py
```

### Configuration

DETR training uses:
- COCO-format annotations
- HuggingFace Transformers (facebook/detr-resnet-50)
- Fine-tuning on custom dataset

#### Default Hyperparameters

```python
# In DETRTrainer
TrainingConfig(
    epochs=20,
    batch_size=4,              # Smaller due to transformer memory
    learning_rate=1e-4,        # Lower LR for transformer fine-tuning
    weight_decay=1e-4,
    lr_backbone=1e-5,          # Even lower for backbone
    image_size=800             # DETR uses larger images
)
```

### Custom Training

```bash
# Train for 150 epochs (recommended for convergence)
CCTV_TRAINING__EPOCHS=150 uv run cctv-train-detr

# Use smaller batch size if out of memory
CCTV_TRAINING__BATCH_SIZE=2 uv run cctv-train-detr

# Adjust learning rates
CCTV_TRAINING__LEARNING_RATE=5e-5 uv run cctv-train-detr
```

### Training Output

Results are saved to `runs/detr/final/`:

```
runs/detr/final/
├── pytorch_model.bin        # Model weights
├── config.json              # Model configuration
├── evaluation_metrics.json  # mAP scores
├── training_history.json    # Loss history
└── preprocessor_config.json # Image preprocessor config
```

### Evaluation

DETR training automatically computes mAP@0.5 and mAP@0.5:0.95 after training.

To re-evaluate:

```bash
uv run cctv-evaluate-detr
```

## Model Comparison

### Convergence Speed

Different architectures require different training durations:

| Model | Architecture | Typical Convergence | Default Epochs | Recommended for Production |
|-------|--------------|---------------------|----------------|---------------------------|
| **YOLOv8** | Single-stage CNN | Fast (20-50) | 20 | 30-50 |
| **Faster R-CNN** | Two-stage CNN | Medium (50-100) | 20 | 50-100 |
| **DETR** | Transformer | Slow (100-300) | 20 | 100-200 |

### Training Efficiency vs Peak Performance

**Default (20 epochs)**: Compares **training efficiency** - which architecture learns fastest with limited training budget.

**Convergence**: For **peak performance**, train each model until convergence:
- YOLOv8: 30-50 epochs
- Faster R-CNN: 50-100 epochs
- DETR: 100-300 epochs

### Resource Requirements

| Model | GPU Memory | Training Time (20 epochs) |
|-------|-----------|---------------------------|
| YOLOv8 | 4-8GB | ~1-2 hours |
| Faster R-CNN | 6-10GB | ~2-3 hours |
| DETR | 8-16GB | ~4-6 hours |

*Times approximate on M1 Pro/NVIDIA RTX 3080*

## Device Selection

Training automatically selects the best available device:

1. **Apple Metal (MPS)**: M1/M2/M3 Macs
2. **CUDA**: NVIDIA GPUs
3. **CPU**: Fallback (much slower)

Force a specific device:

```python
# In training script
device = torch.device("cpu")  # Force CPU
```

## Data Augmentation

### YOLOv8

Built-in augmentations:
- Mosaic (0.8 probability)
- Mixup (0.1 probability)
- Random scaling
- Random horizontal flip
- HSV color jitter

### Faster R-CNN

Custom augmentations in DataLoader:
- Random horizontal flip
- Color jitter
- Normalization

### DETR

HuggingFace transformations:
- Random resize (400-800px)
- Random horizontal flip
- Normalization (ImageNet stats)

## Transfer Learning

All models use transfer learning for faster convergence:

### YOLOv8
- Pre-trained on COCO dataset
- Freezes layers [0, 1, 2, 3] (backbone)
- Fine-tunes detection head

### Faster R-CNN
- Pre-trained ResNet-50 FPN backbone
- Replaces classification head for custom classes
- Fine-tunes entire model

### DETR
- Pre-trained facebook/detr-resnet-50
- Fine-tunes all layers with lower learning rate
- Separate LR for backbone and transformer

## Monitoring Training

### YOLOv8

Watch training in real-time:

```bash
# Results saved to runs/detect/train/
# Monitor results.csv for per-epoch metrics
tail -f runs/detect/train/results.csv
```

### Faster R-CNN

Use TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir runs/faster_rcnn/train/logs

# View at http://localhost:6006
```

### DETR

Check training history:

```bash
# View loss history
cat runs/detr/final/training_history.json
```

## Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce batch size

```bash
CCTV_TRAINING__BATCH_SIZE=4 uv run cctv-train
CCTV_TRAINING__BATCH_SIZE=2 uv run cctv-train-detr
```

### Slow Convergence

**Solution**: Increase learning rate or epochs

```bash
CCTV_TRAINING__LEARNING_RATE=0.01 CCTV_TRAINING__EPOCHS=50 uv run cctv-train
```

### Overfitting

**Solution**: Increase weight decay or reduce model complexity

```bash
CCTV_TRAINING__WEIGHT_DECAY=0.005 uv run cctv-train
```

### Poor mAP Scores

**Solutions**:
1. Train for more epochs
2. Check dataset quality and labeling
3. Adjust confidence thresholds during evaluation
4. Try different learning rates

### Training Crashes

**Solution**: Check logs and ensure dataset is properly formatted

```bash
# Validate dataset structure
ls -R datasets/ultralytics/

# Check data.yaml
cat datasets/ultralytics/data.yaml
```

## Best Practices

1. **Start with YOLOv8**: Fastest training, good baseline performance
2. **Use transfer learning**: All models benefit from pre-trained weights
3. **Monitor validation metrics**: Watch for overfitting
4. **Save checkpoints**: Enable resuming training if interrupted
5. **Validate dataset**: Ensure labels are correct before training
6. **Use GPU**: Training on CPU is very slow
7. **Adjust batch size**: Balance speed vs memory usage
8. **Train longer for DETR**: Transformers need more epochs

## Advanced Topics

### Hyperparameter Tuning

Use grid search or Bayesian optimization:

```bash
# Try different learning rates
for lr in 0.001 0.005 0.01; do
    CCTV_TRAINING__LEARNING_RATE=$lr uv run cctv-train
done
```

### Multi-GPU Training

YOLOv8 supports multi-GPU:

```python
# In training script
model.train(
    data=data_yaml,
    epochs=20,
    device=[0, 1]  # Use GPUs 0 and 1
)
```

### Resume Training

```python
# YOLOv8
model = YOLO('runs/detect/train/weights/last.pt')
model.train(resume=True)
```

### Custom Augmentations

Modify augmentation in trainer classes:

```python
# In FasterRCNNTrainer
def get_transforms(self, train=True):
    transforms = [
        T.ToTensor(),
        T.RandomHorizontalFlip(0.5) if train else None,
        # Add custom augmentations here
    ]
    return T.Compose([t for t in transforms if t])
```

## Next Steps

After training:
1. **Evaluate models**: See [Evaluation Guide](EVALUATION.md)
2. **Compare performance**: Use the Performance Dashboard
3. **Run inference**: See [UI Guide](UI_GUIDE.md)
4. **Benchmark speed**: `uv run cctv-benchmark`

## Further Reading

- [Evaluation Guide](EVALUATION.md) - Model evaluation and comparison
- [Configuration Guide](CONFIGURATION.md) - Detailed configuration options
- [Dataset Guide](DATASET.md) - Dataset preparation and formatting
- [Architecture Guide](ARCHITECTURE.md) - Training architecture details