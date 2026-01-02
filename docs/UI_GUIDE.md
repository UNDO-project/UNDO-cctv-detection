# 🎛️ User Interface Guide

This guide covers using the Gradio web interface for CCTV detection, model comparison, and performance analysis.

## Overview

The CCTV Detection System provides an interactive web interface built with Gradio that allows you to:
- Run object detection with any of the three models
- Compare model predictions side-by-side
- View training metrics and performance dashboards
- Test with example images
- Benchmark inference speed

## Launching the UI

### Start the Web Interface

```bash
# From project root (canonical entry point)
uv run python app.py

# OR use the console script
uv run cctv-ui
```

The interface will launch and display:

```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

Open your browser and navigate to `http://127.0.0.1:7860`.

### Public Access

To create a public shareable link:

```python
# In app.py, modify launch_ui()
demo.launch(share=True)
```

This generates a temporary public URL (valid for 72 hours).

## Interface Overview

The UI consists of five tabs:

1. **🎯 Single Model Detection** - Upload and detect with one model
2. **⚖️ Model Comparison** - Compare all three models side-by-side
3. **🖼️ Example Images** - Pre-loaded test images
4. **📊 Performance Dashboard** - Training metrics and analytics
5. **ℹ️ About** - Model information and citation

## Tab 1: Single Model Detection

### Purpose
Run object detection on uploaded images using a single model with adjustable confidence threshold.

### How to Use

1. **Select Model**: Choose from dropdown
   - YOLOv8 (Fast)
   - Faster R-CNN (Balanced)
   - DETR (Accurate)

2. **Upload Image**:
   - Click or drag image into upload area
   - Supported formats: JPG, PNG, JPEG
   - Recommended size: < 5MB

3. **Adjust Confidence**:
   - Slider range: 0.0 to 1.0
   - Default: 0.25
   - Lower = more detections (more false positives)
   - Higher = fewer detections (more false negatives)

4. **Click "Detect"**

### Output

**Left Panel**: Input image

**Right Panel**: Detection results
- Bounding boxes around detected objects
- Class labels (CCTV or CCTV-SIGNS)
- Confidence scores

**Bottom**: Performance metrics
- Number of detections found
- Inference time (milliseconds)
- Model used

### Example Workflow

```
1. Select "YOLOv8" from dropdown
2. Upload "street_camera.jpg"
3. Set confidence to 0.5
4. Click "Detect"
5. View results: 3 CCTV cameras detected in 42ms
```

## Tab 2: Model Comparison

### Purpose
Run all three models on the same image and compare results side-by-side.

### How to Use

1. **Upload Image**: Single upload for all models

2. **Adjust Confidence**: Single slider applies to all models

3. **Click "Compare Models"**

### Output

Three columns showing:
- **YOLOv8 Results**: Fastest inference
- **Faster R-CNN Results**: Balanced performance
- **DETR Results**: Highest accuracy

Each column displays:
- Annotated image with detections
- Number of detections
- Inference time

### Use Cases

- **Quality Comparison**: Which model finds more cameras?
- **Speed Comparison**: Which model is fastest?
- **Consistency Check**: Do all models agree on detections?
- **Model Selection**: Decide which model works best for your images

### Interpreting Results

**High Agreement**: All models detect same objects → High confidence in detections

**Disagreement**: Different detections → Review manually, possibly ambiguous objects

**Speed Differences**:
- YOLOv8: Typically 40-60ms
- Faster R-CNN: Typically 100-150ms
- DETR: Typically 120-180ms

## Tab 3: Example Images

### Purpose
Quickly test models with pre-loaded validation images without uploading.

### How to Use

1. **Browse Gallery**: Scroll through example images

2. **Click Image**: Loads image into Single Model Detection tab

3. **Run Detection**: Switch to Tab 1 and click "Detect"

### Preparing Examples

Populate the examples gallery:

```bash
# Copy validation images to examples/
uv run cctv-prepare-examples

# Or run directly
uv run python scripts/prepare_examples.py
```

This copies 6 representative images from the validation set to `examples/`.

### Customizing Examples

Add your own images to `examples/` directory:

```bash
# Add custom example images
cp my_test_image.jpg examples/

# UI will automatically load all images in examples/
```

## Tab 4: Performance Dashboard

### Purpose
View training metrics, model comparisons, and performance analytics.

### Components

#### 1. Training Summary Table

Displays key metrics for all trained models:

| Column | Description |
|--------|-------------|
| Model | Model name |
| Epochs | Number of training epochs |
| mAP@0.5 | Mean Average Precision at 50% IoU |
| mAP@0.5:0.95 | Mean AP at 50-95% IoU (COCO metric) |
| Final Loss | Training loss at last epoch |
| Status | Performance indicator |

**Status Indicators:**
- ✅ **Excellent** (mAP@0.5 ≥ 0.7): Production-ready
- ✅ **Good** (mAP@0.5 ≥ 0.5): Acceptable performance
- 🔄 **In Progress** (mAP@0.5 ≥ 0.3): Needs more training
- ⚠️ **Needs Training** (mAP@0.5 < 0.3): Insufficient training

#### 2. mAP Comparison Charts

**Bar Charts** comparing:
- mAP@0.5 across all models
- mAP@0.5:0.95 across all models

Use these to quickly see which model has best accuracy.

#### 3. Training Loss Curves

**Line charts** showing loss over epochs for each model:
- Training loss (solid line)
- Validation loss (dashed line)

**Interpretation:**
- Decreasing curves = Learning
- Converging curves = Near optimal
- Diverging curves = Overfitting

#### 4. Inference Speed Benchmark

**Table** comparing inference speed:

| Model | Avg Time (ms) | Throughput (img/s) |
|-------|---------------|-------------------|
| YOLOv8 | 45 | 22.2 |
| Faster R-CNN | 126 | 7.9 |
| DETR | 156 | 6.4 |

**Use Case:** Determine if model is fast enough for your application.

### When to Check Dashboard

- **After training**: Verify model learned successfully
- **Before deployment**: Compare options for production
- **Debugging**: Identify overfitting or poor performance
- **Reporting**: Export metrics for documentation

## Tab 5: About

### Contents

- **Model Descriptions**: Overview of each architecture
- **Citation Information**: How to cite this work
- **Project Links**: UNDO project webpage, GitHub repo
- **Dataset Information**: Classes, format, access

## Advanced Features

### Batch Processing

The UI processes one image at a time. For batch processing:

```python
# Use detector directly in Python
from src.infrastructure.detector_factory import DetectorFactory

factory = DetectorFactory()
detector = factory.create_detector("yolo")

for image_path in image_list:
    results = detector.detect(image_path, confidence=0.25)
    # Process results
```

### Custom Confidence Thresholds

Find optimal threshold for your use case:

```
1. Upload test image
2. Start with confidence=0.25
3. Increase until false positives disappear
4. Decrease if missing true positives
5. Record optimal value for production use
```

### Exporting Results

Currently, results are displayed but not automatically saved. To save:

**Browser**: Right-click annotated image → "Save Image As"

**Programmatic**:
```python
# Save detection results
detector.detect(image_path, confidence=0.25, save=True)
```

## Performance Tips

### Slow Inference

**Possible Causes:**
- CPU-only mode (no GPU)
- Large image size
- DETR model (inherently slower)

**Solutions:**
- Resize images before upload
- Use YOLOv8 for speed
- Enable GPU/MPS acceleration

### UI Not Loading

**Check:**
1. Port 7860 not in use: `lsof -i :7860`
2. Model weights exist in `samples/`
3. Dependencies installed: `uv sync --all-extras`

### Out of Memory

**Solutions:**
- Close other applications
- Use smaller images
- Reduce batch size if training
- Use CPU mode: Set `device="cpu"` in code

## Customizing the UI

### Changing Port

```python
# In app.py
demo.launch(server_port=8080)
```

### Adding Models

To add a new model to the UI:

1. **Implement detector**: Create detector class in `infrastructure/`
2. **Update factory**: Add to `DetectorFactory`
3. **Update UI**: Add to model dropdown in `gradio_app.py`

```python
# In gradio_app.py
model_choice = gr.Dropdown(
    choices=["YOLOv8", "Faster R-CNN", "DETR", "MyNewModel"],
    value="YOLOv8",
    label="Select Model"
)
```

### Customizing Theme

```python
# In gradio_app.py
demo = gr.Blocks(theme=gr.themes.Soft())  # or Monochrome, Glass, etc.
```

### Adding Custom Tabs

```python
# In gradio_app.py create_demo()
with gr.Blocks() as demo:
    with gr.Tab("My Custom Tab"):
        gr.Markdown("# Custom Content")
        # Add components
```

## Troubleshooting

### Model Weights Not Found

**Error**: "Model weights not found at samples/best.pt"

**Solution**:
1. Download pre-trained weights or train models
2. Ensure weights are in `samples/` directory
3. Check file paths in configuration

### Detection Not Working

**Check:**
1. Image uploaded successfully
2. Confidence threshold not too high (try 0.25)
3. Model weights loaded (check console output)

### Dashboard Shows No Data

**Cause**: Models not yet trained

**Solution**: Train at least one model:
```bash
uv run cctv-train  # Train YOLOv8
```

### Slow UI Response

**Possible Causes:**
- Running on CPU
- Large image size
- Multiple models in comparison mode

**Solutions:**
- Enable GPU/MPS
- Resize images to < 1MB
- Use single model detection instead of comparison

## Keyboard Shortcuts

- **Ctrl+C** (in terminal): Stop UI server
- **Ctrl+R** (in browser): Refresh UI
- **F12**: Open browser dev tools (for debugging)

## Best Practices

1. **Start with examples**: Test with example images before uploading
2. **Use model comparison**: Compare all three models on important images
3. **Optimize confidence**: Find threshold that balances precision and recall
4. **Check dashboard**: Review training metrics before trusting results
5. **Save good examples**: Keep images that work well for documentation

## Integration with Other Tools

### Export to Annotation Tools

Convert detections to Label Studio format:

```python
# Custom export script
detections = detector.detect(image_path)
label_studio_json = convert_to_labelstudio(detections)
```

### API Access

For programmatic access without UI:

```python
from src.infrastructure.detector_factory import DetectorFactory

factory = DetectorFactory()
detector = factory.create_detector("yolo")
results = detector.detect("image.jpg", confidence=0.25)

for detection in results:
    print(f"Class: {detection.class_name}")
    print(f"Confidence: {detection.confidence}")
    print(f"Bbox: {detection.bbox}")
```

## Next Steps

After using the UI:
- **Review metrics**: Check [Evaluation Guide](EVALUATION.md)
- **Improve models**: See [Training Guide](TRAINING.md)
- **Deploy**: Integrate detectors into your application
- **Contribute**: Add features via [Development Guide](DEVELOPMENT.md)

## Further Reading

- [Evaluation Guide](EVALUATION.md) - Understanding performance metrics
- [Training Guide](TRAINING.md) - Improving model accuracy
- [Configuration Guide](CONFIGURATION.md) - Customizing settings
- [Development Guide](DEVELOPMENT.md) - Contributing to the UI