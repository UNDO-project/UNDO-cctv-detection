# 🗂️ Dataset Guide

This guide covers dataset structure, format conversion, image labeling, and dataset preparation for training CCTV detection models.

## Overview

The CCTV Detection System supports multiple dataset formats:
- **YOLO format**: Used by YOLOv8
- **COCO format**: Used by DETR and Faster R-CNN (converted internally)
- **Label Studio format**: For annotation and labeling

## Dataset Classes

The system detects two classes:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | CCTV | Surveillance cameras (domes, bullets, PTZ) |
| 1 | CCTV-SIGNS | Signage indicating surveillance ("You are being recorded", etc.) |

## Dataset Structure

### Raw Dataset

Initial dataset structure before preparation:

```
datasets/
├── images/              # All images (JPG, PNG)
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── labels/              # YOLO format annotations
│   ├── image_001.txt
│   ├── image_002.txt
│   └── ...
├── classes.txt          # Class names
└── notes.json           # Dataset metadata
```

### Prepared for Training

After running preparation scripts:

```
datasets/
├── ultralytics/         # YOLOv8 format
│   ├── images/
│   │   ├── train/       # Training images (70%)
│   │   └── val/         # Validation images (30%)
│   ├── labels/
│   │   ├── train/       # Training labels
│   │   └── val/         # Validation labels
│   └── data.yaml        # Dataset configuration
│
├── coco/                # COCO format (for DETR/Faster R-CNN)
│   ├── annotations/
│   │   ├── train.json
│   │   └── val.json
│   ├── images/
│   │   ├── train/
│   │   └── val/
│
└── labelstudio/         # Label Studio format
    └── annotations.json
```

## Annotation Formats

### YOLO Format

Each image has a corresponding `.txt` file with one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to [0, 1]:

**Example** (`image_001.txt`):
```
0 0.5 0.3 0.15 0.2
1 0.7 0.8 0.1 0.05
```

- Line 1: CCTV camera at center (0.5, 0.3), size 15% × 20%
- Line 2: CCTV sign at (0.7, 0.8), size 10% × 5%

### COCO Format

JSON file with image and annotation metadata:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 300, 400],
      "area": 120000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "CCTV"},
    {"id": 2, "name": "CCTV-SIGNS"}
  ]
}
```

Bboxes are `[x, y, width, height]` in absolute pixels.

## Dataset Preparation

### Splitting Dataset

Split images into train/validation/test sets:

```python
from src.infrastructure.splitters import RandomSplitter

splitter = RandomSplitter()
splitter.split(
    source_dir="datasets/images",
    output_dir="datasets/ultralytics",
    train_ratio=0.7,
    val_ratio=0.3,
    test_ratio=0.0
)
```

**Configuration**:
- Train: 70%
- Validation: 30%
- Test: 0% (optional, use for final evaluation)

### Creating data.yaml

For YOLOv8, create `datasets/ultralytics/data.yaml`:

```yaml
# Dataset paths
path: /absolute/path/to/datasets/ultralytics
train: images/train
val: images/val

# Classes
nc: 2  # Number of classes
names: ['CCTV', 'CCTV-SIGNS']
```

**Important**: Use absolute paths for `path`.

### Converting to COCO Format

For DETR and Faster R-CNN:

```python
from src.infrastructure.dataset_preparer_impl import DatasetPreparerImpl

preparer = DatasetPreparerImpl()
preparer.convert_yolo_to_coco(
    yolo_dir="datasets/ultralytics",
    output_dir="datasets/coco"
)
```

## Image Labeling

### Using Label Studio

Label Studio provides a web-based annotation interface.

#### Setup

1. **Pull Docker image**:
   ```bash
   docker pull heartexlabs/label-studio:latest
   ```

2. **Run container**:
   ```bash
   docker run -it -p 8080:8080 \
     -v $(pwd)/mydata:/label-studio/data \
     heartexlabs/label-studio:latest
   ```

3. **Access interface**:
   - Open `http://localhost:8080`
   - Create username and password

#### Labeling Workflow

1. **Create Project**:
   - Click "Create Project"
   - Name: "CCTV Detection"
   - Import images from `datasets/images/`

2. **Configure Labeling Interface**:
   ```xml
   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="CCTV" background="red"/>
       <Label value="CCTV-SIGNS" background="blue"/>
     </RectangleLabels>
   </View>
   ```

3. **Label Images**:
   - Draw bounding boxes around cameras and signs
   - Assign correct class label
   - Submit annotation

4. **Export Annotations**:
   - Export → YOLO format
   - Save to `datasets/labels/`

### Labeling Best Practices

1. **Tight bounding boxes**: Box should closely fit the object
2. **Include entire object**: Don't cut off parts
3. **Consistent labeling**: Use same criteria for all images
4. **Multiple annotators**: Have 2-3 people label for quality
5. **Difficult cases**: Skip very small or ambiguous objects

### Annotation Guidelines

#### CCTV (Class 0)

**Include**:
- Dome cameras
- Bullet cameras
- PTZ cameras
- Visible camera housings
- Camera bodies with lenses

**Exclude**:
- Dummy/fake cameras (if identifiable)
- Camera icons in signs
- Reflections

#### CCTV-SIGNS (Class 1)

**Include**:
- "You are being recorded" signs
- "Video surveillance" signs
- Camera warning pictograms
- Security monitoring notices

**Exclude**:
- Security company logos
- General warning signs without camera reference

## Dataset Quality

### Validation Checks

Before training, validate dataset quality:

```python
from src.infrastructure.dataset_preparer_impl import DatasetValidator

validator = DatasetValidator()
report = validator.validate(
    images_dir="datasets/ultralytics/images/train",
    labels_dir="datasets/ultralytics/labels/train"
)

print(report)
# {
#   "total_images": 500,
#   "total_annotations": 1234,
#   "images_without_labels": 0,
#   "labels_without_images": 0,
#   "invalid_bboxes": 0,
#   "class_distribution": {"CCTV": 1000, "CCTV-SIGNS": 234}
# }
```

### Common Issues

#### Missing Labels

**Problem**: Images without corresponding label files

**Solution**:
```bash
# Find images without labels
find datasets/images -name "*.jpg" | while read img; do
  label="${img%.jpg}.txt"
  label="${label/images/labels}"
  [ -f "$label" ] || echo "Missing: $label"
done
```

#### Class Imbalance

**Problem**: Many CCTV cameras, few CCTV signs

**Solution**:
- Collect more images with signs
- Use data augmentation
- Adjust class weights during training

#### Invalid Bounding Boxes

**Problem**: Coordinates outside [0, 1] range

**Solution**:
```python
# Fix invalid coordinates
def fix_bbox(bbox):
    return [max(0.0, min(1.0, coord)) for coord in bbox]
```

## Data Augmentation

### YOLOv8 Built-in Augmentation

YOLOv8 automatically applies:
- Mosaic (combines 4 images)
- Mixup (blends 2 images)
- Random scaling
- Random flipping
- HSV color jitter

### Custom Augmentation

For additional augmentation:

```python
import albumentations as A

transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Blur(blur_limit=3, p=0.2)
], bbox_params=A.BboxParams(format='yolo'))
```

## Dataset Statistics

### Current Dataset

- **Total images**: ~500 (varies by version)
- **Train/Val split**: 70/30
- **Classes**: 2 (CCTV, CCTV-SIGNS)
- **Average objects per image**: 2.5
- **Image resolution**: 640×640 to 1920×1080

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| CCTV | ~1000 | ~80% |
| CCTV-SIGNS | ~250 | ~20% |

### Recommended Dataset Size

For good performance:
- **Minimum**: 200 images, 500 annotations
- **Good**: 500 images, 1500 annotations
- **Excellent**: 1000+ images, 3000+ annotations

## Collecting New Data

### Web Scraping

Use the built-in image scraper:

```python
from src.infrastructure.image_scraper import ImageScraper

scraper = ImageScraper()
scraper.scrape(
    url="https://example.com/cameras",
    output_dir="datasets/scraped",
    max_images=100
)
```

### Manual Collection

1. **Search image databases**:
   - Google Images (check licenses)
   - Flickr (CC-licensed images)
   - Pexels/Unsplash (free stock photos)

2. **Capture own images**:
   - Photograph public CCTV cameras
   - Ensure legal compliance (public spaces)
   - Respect privacy laws

3. **Use existing datasets**:
   - [Fuziih CCTV-Exposure](https://github.com/Fuziih/cctv-exposure)
   - Other public surveillance datasets

## Data Sources

### Current Dataset Sources

1. **Ethnographic research**: Custom collected images
2. **Fuziih CCTV-Exposure**: Public dataset
3. **Web scraping**: Legally obtained images

### Licensing

- **This dataset**: Contact UNDO project for access
- **CC0/Public Domain**: Include attribution
- **CC-BY**: Provide attribution
- **Commercial use**: Verify license compatibility

## Format Conversion

### YOLO → COCO

```python
from src.infrastructure.dataset_preparer_impl import YoloToCocoConverter

converter = YoloToCocoConverter()
converter.convert(
    yolo_images_dir="datasets/ultralytics/images/train",
    yolo_labels_dir="datasets/ultralytics/labels/train",
    output_json="datasets/coco/annotations/train.json"
)
```

### COCO → YOLO

```python
from src.infrastructure.dataset_preparer_impl import CocoToYoloConverter

converter = CocoToYoloConverter()
converter.convert(
    coco_json="datasets/coco/annotations/train.json",
    output_dir="datasets/ultralytics"
)
```

### Label Studio → YOLO

Export from Label Studio:
1. Project → Export
2. Select "YOLO" format
3. Download and extract to `datasets/labels/`

## Dataset Versioning

### Track Dataset Changes

Use git-lfs for large files:

```bash
# Install git-lfs
git lfs install

# Track images
git lfs track "datasets/**/*.jpg"
git lfs track "datasets/**/*.png"

# Commit
git add .gitattributes
git add datasets/
git commit -m "Add dataset v1.0"
```

### Dataset Changelog

Document changes in `datasets/CHANGELOG.md`:

```markdown
# Dataset Changelog

## v1.1 (2024-02-01)
- Added 100 new images with CCTV signs
- Fixed 15 incorrect labels
- Balanced class distribution

## v1.0 (2024-01-01)
- Initial dataset release
- 500 images, 1250 annotations
```

## Troubleshooting

### Label Studio Not Starting

**Solution**: Check port 8080 is available

```bash
lsof -i :8080
# Kill process if needed
```

### Format Conversion Errors

**Solution**: Validate input format first

```bash
# Check YOLO format
head -n 5 datasets/labels/image_001.txt

# Verify all values in [0, 1]
```

### Missing Images After Split

**Solution**: Check source directory and file permissions

```bash
ls -la datasets/images/
```

## Best Practices

1. **Backup raw data**: Keep original images and labels
2. **Version datasets**: Track changes with git or DVC
3. **Validate before training**: Run validation scripts
4. **Document sources**: Record where images came from
5. **Balance classes**: Aim for roughly equal class distribution
6. **Quality over quantity**: 500 good labels > 1000 poor labels
7. **Regular audits**: Review random samples for quality

## Next Steps

After preparing the dataset:
1. **Train models**: See [Training Guide](TRAINING.md)
2. **Evaluate quality**: Check [Evaluation Guide](EVALUATION.md)
3. **Iterate**: Collect more data for poorly performing classes

## Further Reading

- [Training Guide](TRAINING.md) - Using prepared datasets for training
- [Evaluation Guide](EVALUATION.md) - Assessing dataset quality through metrics
- [Configuration Guide](CONFIGURATION.md) - Dataset path configuration
- [Label Studio Documentation](https://labelstud.io/guide/)