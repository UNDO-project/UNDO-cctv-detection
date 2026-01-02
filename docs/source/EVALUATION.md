# 📈 Evaluation Guide

This guide covers model evaluation, performance comparison, and the methodology for comparing different object detection architectures.

## Overview

The CCTV Detection System provides comprehensive evaluation tools to:
- Compute mAP (mean Average Precision) metrics
- Compare performance across three architectures
- Benchmark inference speed
- Visualize training progress and results

## Evaluation Metrics

### Mean Average Precision (mAP)

The primary evaluation metric is **mAP** (mean Average Precision), computed at different IoU (Intersection over Union) thresholds:

#### mAP@0.5
- Average precision at 50% IoU threshold
- Standard metric for object detection
- More lenient (accepts loose bounding boxes)

#### mAP@0.5:0.95
- Average precision across IoU thresholds from 0.5 to 0.95 (step 0.05)
- COCO evaluation standard
- More strict (requires precise bounding boxes)

### Other Metrics

- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Inference Time**: Time to process a single image (milliseconds)

## Evaluating Models

### YOLOv8 Evaluation

YOLOv8 automatically computes metrics during training:

```bash
# Metrics saved to runs/detect/train/results.csv
cat runs/detect/train/results.csv
```

Manual evaluation:

```python
from ultralytics import YOLO

model = YOLO('samples/best.pt')
metrics = model.val(data='datasets/ultralytics/data.yaml')

print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")
```

### Faster R-CNN Evaluation

Evaluate trained Faster R-CNN model:

```bash
# Computes mAP using COCO evaluation protocol
uv run cctv-evaluate-faster-rcnn

# Or run directly
uv run python scripts/evaluate_faster_rcnn.py
```

Output saved to:
- `runs/faster_rcnn/train/evaluation_metrics.json`
- `runs/faster_rcnn/train/training_metrics.json`

### DETR Evaluation

Evaluate trained DETR model:

```bash
# Computes mAP using COCO evaluation protocol
uv run cctv-evaluate-detr

# Or run directly
uv run python scripts/evaluate_detr.py
```

Output saved to:
- `runs/detr/final/evaluation_metrics.json`

## Model Comparison Methodology

### Fair Comparison Principles

When comparing models, consider both **training efficiency** and **peak performance**:

#### 1. Equal Epochs Comparison (Default)

Training all models for the same number of epochs (20) shows **training efficiency**:

| Model | Epochs | mAP@0.5 | mAP@0.5:0.95 | Interpretation |
|-------|--------|---------|--------------|----------------|
| YOLOv8 | 20 | 0.873 | 0.415 | Good performance, near convergence |
| Faster R-CNN | 20 | ~0.6-0.7 | ~0.3-0.4 | Still learning, needs more epochs |
| DETR | 20 | ~0.3-0.5 | ~0.2-0.3 | Early stage, needs many more epochs |

**Use Case**: Quick prototyping, limited compute budget, sample efficiency analysis

#### 2. Convergence Comparison

Training each model until convergence shows **peak performance**:

| Model | Epochs to Converge | Expected mAP@0.5 | Training Time |
|-------|-------------------|------------------|---------------|
| YOLOv8 | 30-50 | 0.85-0.90 | 2-4 hours |
| Faster R-CNN | 50-100 | 0.80-0.88 | 4-8 hours |
| DETR | 100-300 | 0.75-0.85 | 10-30 hours |

**Use Case**: Production deployment, best possible accuracy, final model selection

### Convergence Speed vs Final Performance

Different architectures have different learning curves:

```
Performance
    │
    │     ┌── YOLOv8 (fast convergence)
    │    ╱
    │   ╱  ┌── Faster R-CNN (medium)
    │  ╱  ╱
    │ ╱  ╱  ┌── DETR (slow convergence)
    │╱  ╱  ╱
    │  ╱  ╱
    │ ╱  ╱
    │╱  ╱
    └────────────────────→ Training Epochs
    0   20   50   100  300
```

**Key Insight**: DETR may underperform CNNs at 20 epochs but can match or exceed them with sufficient training (100+ epochs).

## Performance Dashboard

The web UI includes a comprehensive Performance Dashboard showing:

### 1. Training Summary Table

| Model | Epochs | mAP@0.5 | mAP@0.5:0.95 | Final Loss | Status |
|-------|--------|---------|--------------|------------|--------|
| YOLOv8 | 20 | 0.873 | 0.415 | 5.0 | ✅ Excellent |
| Faster R-CNN | 20 | 0.650 | 0.320 | 0.45 | ✅ Good |
| DETR | 20 | 0.420 | 0.250 | 1.2 | 🔄 In Progress |

#### Status Indicators

- **✅ Excellent** (mAP@0.5 ≥ 0.7): Production-ready performance
- **✅ Good** (mAP@0.5 ≥ 0.5): Acceptable performance
- **🔄 In Progress** (mAP@0.5 ≥ 0.3): Learning but needs more training
- **⚠️ Needs Training** (mAP@0.5 < 0.3): Very early stage or insufficient training

### 2. mAP Comparison Charts

Interactive charts comparing:
- mAP@0.5 across all models (bar chart)
- mAP@0.5:0.95 across all models (bar chart)

### 3. Training History

Loss curves for each model:
- Training loss over epochs
- Validation loss over epochs
- Identifies overfitting (diverging curves)

### 4. Inference Benchmark

Real-time speed comparison:
- Average inference time per image (ms)
- Throughput (images/second)
- Device information (MPS/CUDA/CPU)

## Benchmarking Inference Speed

Run comprehensive speed benchmark:

```bash
uv run cctv-benchmark

# Or run directly
uv run python scripts/benchmark.py
```

Output:

```
Model Performance Benchmark
==========================

YOLOv8:
  Average inference time: 45.2 ms
  Throughput: 22.1 images/sec

Faster R-CNN:
  Average inference time: 125.7 ms
  Throughput: 8.0 images/sec

DETR:
  Average inference time: 156.3 ms
  Throughput: 6.4 images/sec
```

## Interpreting Results

### When to Choose Each Model

#### Choose YOLOv8 if:
- Speed is critical (real-time applications)
- Limited training time/compute budget
- Good accuracy is sufficient (not necessarily best)
- Quick iteration is important

#### Choose Faster R-CNN if:
- Accuracy is more important than speed
- Willing to train for 50-100 epochs
- Need good balance of speed and accuracy
- Two-stage detection benefits your use case

#### Choose DETR if:
- Willing to train for 100-300 epochs
- Want cutting-edge transformer architecture
- Accuracy is paramount
- Research or experimentation focus

### Performance vs Speed Tradeoff

```
Accuracy
    │
    │    Faster R-CNN (balanced)
    │         ●
    │    YOLOv8     DETR (slow but accurate)
    │      ●          ●
    │
    │
    │
    └──────────────────────→ Speed
         Fast        Slow
```

## COCO Evaluation Protocol

Both Faster R-CNN and DETR use the COCO evaluation protocol:

### How It Works

1. **IoU Calculation**: Compute Intersection over Union for each detection
2. **Matching**: Match predictions to ground truth boxes
3. **Precision-Recall**: Compute PR curve at each IoU threshold
4. **Average Precision**: Area under PR curve
5. **Mean AP**: Average across all classes and IoU thresholds

### Implementation

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load ground truth and predictions
coco_gt = COCO(gt_annotations)
coco_dt = coco_gt.loadRes(predictions)

# Evaluate
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extract metrics
mAP_50 = coco_eval.stats[1]  # mAP@0.5
mAP_50_95 = coco_eval.stats[0]  # mAP@0.5:0.95
```

## Validation Strategies

### Hold-Out Validation (Default)

Dataset split:
- 70% training
- 30% validation
- Validation set never seen during training

### Cross-Validation

For small datasets, use k-fold cross-validation:

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)
for train_idx, val_idx in kfold.split(dataset):
    # Train on train_idx, validate on val_idx
    pass
```

### Test Set Evaluation

For final model evaluation, use a separate test set:

```
Dataset:
├── Train (60%)
├── Validation (20%)  # Used during training
└── Test (20%)        # Final evaluation only
```

## Common Evaluation Pitfalls

### 1. Data Leakage

**Problem**: Validation images in training set

**Solution**: Ensure strict train/val split before any preprocessing

### 2. Imbalanced Classes

**Problem**: Many CCTV cameras, few CCTV signs

**Solution**: Report per-class AP, not just mean AP

### 3. Threshold Sensitivity

**Problem**: Results vary significantly with confidence threshold

**Solution**: Report mAP across multiple thresholds, use standard 0.5

### 4. Comparing Apples to Oranges

**Problem**: Comparing models trained for different epochs

**Solution**: Either train all to convergence OR explicitly state training budget comparison

### 5. Ignoring Inference Speed

**Problem**: Only considering accuracy

**Solution**: Include speed benchmarks in model selection

## Reproducibility

### Ensuring Reproducible Results

1. **Fix random seeds**:
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   ```

2. **Document configuration**:
   - Model architecture and weights
   - Training hyperparameters
   - Dataset version and split
   - Hardware used

3. **Save evaluation scripts**: Keep evaluation code versioned

4. **Log everything**: Training logs, validation metrics, system info

## Reporting Results

### Academic/Research Reporting

Include:
- Model architecture details
- Training hyperparameters
- Dataset statistics (# images, # instances per class)
- mAP@0.5 and mAP@0.5:0.95
- Inference speed on specified hardware
- Confidence thresholds used

Example:

```
YOLOv8 Results:
- Architecture: YOLOv8n (nano)
- Training: 20 epochs, batch size 8, lr 0.005
- Dataset: 500 images (70/30 train/val split)
- mAP@0.5: 0.873
- mAP@0.5:0.95: 0.415
- Inference: 45ms on M1 Pro (MPS)
- Confidence threshold: 0.25
```

### Production Reporting

Include:
- Model accuracy (mAP@0.5)
- Latency (p50, p95, p99)
- Throughput (images/second)
- Resource usage (GPU memory, CPU)
- Error rates and failure modes

## Advanced Topics

### Confidence Threshold Optimization

Find optimal threshold:

```python
thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
for threshold in thresholds:
    results = detector.detect(image, confidence=threshold)
    # Evaluate precision/recall
```

### Ensemble Methods

Combine multiple models:

```python
# Weighted ensemble
yolo_weight = 0.5
frcnn_weight = 0.3
detr_weight = 0.2

final_predictions = combine_predictions(
    yolo_results, frcnn_results, detr_results,
    weights=[yolo_weight, frcnn_weight, detr_weight]
)
```

### Per-Class Analysis

Analyze performance per class:

```python
# Check if model struggles with CCTV-SIGNS
ap_cctv = metrics['AP_CCTV']
ap_signs = metrics['AP_CCTV_SIGNS']

if ap_signs < 0.5:
    print("Model struggles with CCTV signs - collect more data")
```

## Next Steps

After evaluation:
1. **Compare in UI**: Use Performance Dashboard for visual comparison
2. **Optimize thresholds**: Find best confidence threshold for your use case
3. **Error analysis**: Inspect false positives/negatives
4. **Iterate**: Collect more data for weak classes, retrain

## Further Reading

- [Training Guide](TRAINING.md) - Improve model performance through better training
- [UI Guide](UI_GUIDE.md) - Use the Performance Dashboard
- [Dataset Guide](DATASET.md) - Improve dataset quality
- [Configuration Guide](CONFIGURATION.md) - Adjust evaluation parameters