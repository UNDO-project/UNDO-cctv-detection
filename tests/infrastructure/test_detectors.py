"""Unit tests for object detection wrappers and factory.

Tests the unified ObjectDetector interface implementations for YOLO,
DETR, and Faster R-CNN models, as well as the DetectorFactory.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.infrastructure.detector_factory import DetectorFactory
from src.infrastructure.detr_detector import DETRDetector
from src.infrastructure.faster_rcnn_detector import FasterRCNNDetector
from src.infrastructure.yolo_detector import YOLODetector


@pytest.fixture
def sample_image():
    """Create a simple test image.

    :return: Random RGB PIL image
    :rtype: Image.Image
    """
    return Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))


@pytest.fixture
def class_names():
    """Standard class names for CCTV detection.

    :return: List of class names
    :rtype: list[str]
    """
    return ["CCTV", "CCTV-SIGNS"]


class TestYOLODetector:
    """Test cases for YOLODetector class."""

    @patch("src.infrastructure.yolo_detector.YOLO")
    def test_initialization(self, mock_yolo, tmp_path: Path, class_names: list[str]):
        """Test that YOLO detector initializes correctly."""
        model_path = tmp_path / "yolo_model.pt"
        detector = YOLODetector(model_path, class_names)

        assert detector.class_names == class_names
        mock_yolo.assert_called_once_with(str(model_path))

    @patch("src.infrastructure.yolo_detector.YOLO")
    def test_predict(
        self,
        mock_yolo,
        tmp_path: Path,
        class_names: list[str],
        sample_image: Image.Image,
    ):
        """Test YOLO predict method returns correct format."""
        # Setup mock YOLO results
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 20.0, 100.0, 200.0])]
        mock_box.cls = [torch.tensor(0)]
        mock_box.conf = [torch.tensor(0.85)]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance

        # Create detector and predict
        model_path = tmp_path / "yolo_model.pt"
        detector = YOLODetector(model_path, class_names)
        detections = detector.predict(sample_image, confidence_threshold=0.25)

        # Verify results
        assert len(detections) == 1
        assert detections[0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
        assert detections[0]["class_id"] == 0
        assert detections[0]["class_name"] == "CCTV"
        assert detections[0]["confidence"] == pytest.approx(0.85)

    @patch("src.infrastructure.yolo_detector.YOLO")
    def test_annotate_image(
        self,
        mock_yolo,
        tmp_path: Path,
        class_names: list[str],
        sample_image: Image.Image,
    ):
        """Test YOLO annotate_image returns PIL Image."""
        mock_result = MagicMock()
        mock_result.plot.return_value = np.random.randint(
            0, 255, (640, 640, 3), dtype=np.uint8
        )

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance

        model_path = tmp_path / "yolo_model.pt"
        detector = YOLODetector(model_path, class_names)
        annotated = detector.annotate_image(sample_image, confidence_threshold=0.25)

        assert isinstance(annotated, Image.Image)
        mock_yolo_instance.predict.assert_called_once()


class TestDETRDetector:
    """Test cases for DETRDetector class."""

    @patch("src.infrastructure.detr_detector.DetrForObjectDetection")
    @patch("src.infrastructure.detr_detector.DetrImageProcessor")
    def test_initialization(
        self, mock_processor, mock_model, tmp_path: Path, class_names: list[str]
    ):
        """Test that DETR detector initializes correctly."""
        model_path = tmp_path / "detr_model"
        model_path.mkdir()

        detector = DETRDetector(model_path, class_names, device=torch.device("cpu"))

        assert detector.class_names == class_names
        assert detector.device == torch.device("cpu")
        mock_processor.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once_with(str(model_path))

    @patch("src.infrastructure.detr_detector.DetrForObjectDetection")
    @patch("src.infrastructure.detr_detector.DetrImageProcessor")
    def test_predict(
        self,
        mock_processor_class,
        mock_model_class,
        tmp_path: Path,
        class_names: list[str],
        sample_image: Image.Image,
    ):
        """Test DETR predict method returns correct format."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "pixel_values": torch.randn(1, 3, 800, 800),
        }
        mock_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.9, 0.75]),
                "labels": torch.tensor([0, 1]),
                "boxes": torch.tensor(
                    [[10.0, 20.0, 100.0, 200.0], [50.0, 60.0, 150.0, 250.0]]
                ),
            }
        ]
        mock_processor_class.from_pretrained.return_value = mock_processor

        # Setup mock model
        mock_model = MagicMock()
        mock_model.return_value = {"logits": torch.randn(1, 100, 3)}
        mock_model_class.from_pretrained.return_value = mock_model

        # Create detector and predict
        model_path = tmp_path / "detr_model"
        model_path.mkdir()
        detector = DETRDetector(model_path, class_names, device=torch.device("cpu"))
        detections = detector.predict(sample_image, confidence_threshold=0.7)

        # Verify results
        assert len(detections) == 2
        assert detections[0]["class_id"] == 0
        assert detections[0]["class_name"] == "CCTV"
        assert detections[1]["class_id"] == 1
        assert detections[1]["class_name"] == "CCTV-SIGNS"

    @patch("src.infrastructure.detr_detector.DetrForObjectDetection")
    @patch("src.infrastructure.detr_detector.DetrImageProcessor")
    def test_annotate_image(
        self,
        mock_processor_class,
        mock_model_class,
        tmp_path: Path,
        class_names: list[str],
        sample_image: Image.Image,
    ):
        """Test DETR annotate_image returns PIL Image."""
        # Setup mocks to return empty detections
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 800, 800)}
        mock_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([]),
                "labels": torch.tensor([]),
                "boxes": torch.tensor([]),
            }
        ]
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model.return_value = {"logits": torch.randn(1, 100, 3)}
        mock_model_class.from_pretrained.return_value = mock_model

        model_path = tmp_path / "detr_model"
        model_path.mkdir()
        detector = DETRDetector(model_path, class_names, device=torch.device("cpu"))
        annotated = detector.annotate_image(sample_image, confidence_threshold=0.7)

        assert isinstance(annotated, Image.Image)


class TestFasterRCNNDetector:
    """Test cases for FasterRCNNDetector class."""

    @patch("src.infrastructure.faster_rcnn_detector.fasterrcnn_resnet50_fpn")
    def test_initialization(
        self, mock_fasterrcnn, tmp_path: Path, class_names: list[str]
    ):
        """Test that Faster R-CNN detector initializes correctly."""
        model_path = tmp_path / "fasterrcnn_model.pt"

        detector = FasterRCNNDetector(
            model_path, class_names, device=torch.device("cpu")
        )

        assert detector.class_names == class_names
        assert detector.device == torch.device("cpu")
        assert detector.num_classes == 3  # 2 classes + 1 background

    @patch("src.infrastructure.faster_rcnn_detector.fasterrcnn_resnet50_fpn")
    def test_predict(
        self,
        mock_fasterrcnn,
        tmp_path: Path,
        class_names: list[str],
        sample_image: Image.Image,
    ):
        """Test Faster R-CNN predict method returns correct format."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.return_value = [
            {
                "boxes": torch.tensor(
                    [[10.0, 20.0, 100.0, 200.0], [50.0, 60.0, 150.0, 250.0]]
                ),
                "labels": torch.tensor([1, 2]),  # +1 for background class
                "scores": torch.tensor([0.9, 0.75]),
            }
        ]
        mock_fasterrcnn.return_value = mock_model

        model_path = tmp_path / "fasterrcnn_model.pt"
        detector = FasterRCNNDetector(
            model_path, class_names, device=torch.device("cpu")
        )
        detections = detector.predict(sample_image, confidence_threshold=0.25)

        # Verify results
        assert len(detections) == 2
        assert detections[0]["class_id"] == 0  # label 1 - 1 = class_id 0
        assert detections[0]["class_name"] == "CCTV"
        assert detections[1]["class_id"] == 1
        assert detections[1]["class_name"] == "CCTV-SIGNS"

    @patch("src.infrastructure.faster_rcnn_detector.fasterrcnn_resnet50_fpn")
    def test_annotate_image(
        self,
        mock_fasterrcnn,
        tmp_path: Path,
        class_names: list[str],
        sample_image: Image.Image,
    ):
        """Test Faster R-CNN annotate_image returns PIL Image."""
        # Setup mock model with no detections
        mock_model = MagicMock()
        mock_model.return_value = [
            {
                "boxes": torch.tensor([]),
                "labels": torch.tensor([]),
                "scores": torch.tensor([]),
            }
        ]
        mock_fasterrcnn.return_value = mock_model

        model_path = tmp_path / "fasterrcnn_model.pt"
        detector = FasterRCNNDetector(
            model_path, class_names, device=torch.device("cpu")
        )
        annotated = detector.annotate_image(sample_image, confidence_threshold=0.25)

        assert isinstance(annotated, Image.Image)


class TestDetectorFactory:
    """Test cases for DetectorFactory class."""

    @patch("src.infrastructure.yolo_detector.YOLO")
    def test_creates_yolo_detector(
        self, mock_yolo, tmp_path: Path, class_names: list[str]
    ):
        """Test factory creates YOLO detector."""
        model_path = tmp_path / "yolo_model.pt"
        detector = DetectorFactory.create_detector(
            model_type="yolo",
            model_path=model_path,
            class_names=class_names,
            device=torch.device("cpu"),
        )
        assert isinstance(detector, YOLODetector)

    @patch("src.infrastructure.detr_detector.DetrForObjectDetection")
    @patch("src.infrastructure.detr_detector.DetrImageProcessor")
    def test_creates_detr_detector(
        self, mock_processor, mock_model, tmp_path: Path, class_names: list[str]
    ):
        """Test factory creates DETR detector."""
        model_path = tmp_path / "detr_model"
        model_path.mkdir()
        detector = DetectorFactory.create_detector(
            model_type="detr",
            model_path=model_path,
            class_names=class_names,
            device=torch.device("cpu"),
        )
        assert isinstance(detector, DETRDetector)

    @patch("src.infrastructure.faster_rcnn_detector.fasterrcnn_resnet50_fpn")
    def test_creates_faster_rcnn_detector(
        self, mock_fasterrcnn, tmp_path: Path, class_names: list[str]
    ):
        """Test factory creates Faster R-CNN detector."""
        model_path = tmp_path / "fasterrcnn_model.pt"
        detector = DetectorFactory.create_detector(
            model_type="faster-rcnn",
            model_path=model_path,
            class_names=class_names,
            device=torch.device("cpu"),
        )
        assert isinstance(detector, FasterRCNNDetector)

    def test_raises_error_for_unknown_model_type(
        self, tmp_path: Path, class_names: list[str]
    ):
        """Test factory raises ValueError for unknown model type."""
        model_path = tmp_path / "model.pt"
        with pytest.raises(ValueError, match="Unknown model type"):
            DetectorFactory.create_detector(
                model_type="unknown",  # type: ignore
                model_path=model_path,
                class_names=class_names,
                device=torch.device("cpu"),
            )

    @patch("src.infrastructure.yolo_detector.YOLO")
    def test_uses_cpu_by_default(
        self, mock_yolo, tmp_path: Path, class_names: list[str]
    ):
        """Test factory defaults to CPU device when device is None."""
        model_path = tmp_path / "yolo_model.pt"
        detector = DetectorFactory.create_detector(
            model_type="yolo",
            model_path=model_path,
            class_names=class_names,
            device=None,
        )
        assert isinstance(detector, YOLODetector)
