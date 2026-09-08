"""Unit tests for gradio_app module.

This module tests the Gradio UI for CCTV detection.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class TestCreateDemo:
    """Test create_demo function."""

    @patch("src.ui.gradio_app.CCTVDetectionApp")
    def test_create_demo_returns_gradio_blocks(self, mock_app_class):
        """Test that create_demo returns a Gradio Blocks object.

        :param mock_app_class: Mocked CCTVDetectionApp class
        :return: None
        """
        from src.ui.gradio_app import create_demo

        # Mock the app instance
        mock_app = MagicMock()
        mock_app.detectors = {"YOLOv8": MagicMock()}
        mock_app_class.return_value = mock_app

        demo = create_demo()

        # Test that demo was created successfully
        assert demo is not None
        # Gradio Blocks objects have a launch method
        assert hasattr(demo, "launch")

    @patch("src.ui.gradio_app.CCTVDetectionApp")
    def test_create_demo_initializes_app(self, mock_app_class):
        """Test that create_demo initializes CCTVDetectionApp.

        :param mock_app_class: Mocked CCTVDetectionApp class
        :return: None
        """
        from src.ui.gradio_app import create_demo

        # Mock the app instance
        mock_app = MagicMock()
        mock_app.detectors = {"YOLOv8": MagicMock()}
        mock_app_class.return_value = mock_app

        create_demo()

        # Verify CCTVDetectionApp was instantiated
        mock_app_class.assert_called_once()


class TestLaunchUI:
    """Test launch_ui function."""

    @patch("src.ui.gradio_app.create_demo")
    def test_launch_ui_creates_and_launches_demo(self, mock_create_demo):
        """Test that launch_ui creates and launches the demo.

        :param mock_create_demo: Mocked create_demo function
        :return: None
        """
        from src.ui.gradio_app import launch_ui

        # Mock the demo instance
        mock_demo = MagicMock()
        mock_create_demo.return_value = mock_demo

        launch_ui()

        # Verify create_demo was called
        mock_create_demo.assert_called_once()
        # Verify demo.launch was called
        mock_demo.launch.assert_called_once()


def _app_with_detectors(detectors: dict):
    """Build a CCTVDetectionApp without loading any model weights.

    :param detectors: Mapping of model name to detector (or None)
    :return: App instance with the given detectors installed
    :rtype: CCTVDetectionApp
    """
    from src.ui.gradio_app import CCTVDetectionApp

    app = CCTVDetectionApp.__new__(CCTVDetectionApp)
    app.detectors = detectors
    app.class_names = ["CCTV", "CCTV-SIGNS"]
    return app


def _detection(class_name: str, confidence: float) -> dict:
    """Build a detection dict in the shape ObjectDetector.predict returns."""
    return {
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "class_id": 0 if class_name == "CCTV" else 1,
        "class_name": class_name,
        "confidence": confidence,
    }


class TestFormatMetrics:
    """_format_metrics builds the per-model Markdown summary."""

    def test_counts_each_class_and_lists_detections(self):
        """Camera and sign counts are separated and every detection is listed."""
        from src.ui.gradio_app import CCTVDetectionApp

        detections = [
            _detection("CCTV", 0.91),
            _detection("CCTV-SIGNS", 0.5),
            _detection("CCTV", 0.75),
        ]

        text = CCTVDetectionApp._format_metrics("YOLOv8", detections, 0.1234)

        assert "### YOLOv8 Results" in text
        assert "| **Inference Time** | 0.123s |" in text
        assert "| **Total Detections** | 3 |" in text
        assert "| **CCTV Cameras** | 2 |" in text
        assert "| **CCTV Signs** | 1 |" in text
        assert "1. **CCTV** - Confidence: 91.00%" in text
        assert "2. **CCTV-SIGNS** - Confidence: 50.00%" in text
        assert "3. **CCTV** - Confidence: 75.00%" in text

    def test_no_detections_reports_zero_counts(self):
        """An empty detection list produces zero counts and no list entries."""
        from src.ui.gradio_app import CCTVDetectionApp

        text = CCTVDetectionApp._format_metrics("DETR", [], 0.0)

        assert "| **Total Detections** | 0 |" in text
        assert "| **CCTV Cameras** | 0 |" in text
        assert "| **CCTV Signs** | 0 |" in text
        assert "Confidence:" not in text


class TestFormatComparisonMetrics:
    """_format_comparison_metrics builds the three-model comparison table."""

    def test_rows_are_in_fixed_model_order_with_counts(self):
        """Each model gets one row with time, total and per-class counts."""
        from src.ui.gradio_app import CCTVDetectionApp

        results = {
            "DETR": (None, [_detection("CCTV-SIGNS", 0.8)], 0.5),
            "YOLOv8": (None, [_detection("CCTV", 0.9), _detection("CCTV", 0.8)], 0.05),
            "Faster R-CNN": (None, [], 0.2),
        }

        text = CCTVDetectionApp._format_comparison_metrics(results)

        rows = [
            line
            for line in text.splitlines()
            if line.startswith("| ") and "Model" not in line
        ]
        assert rows == [
            "| YOLOv8 | 0.050s | 2 | 2 | 0 |",
            "| Faster R-CNN | 0.200s | 0 | 0 | 0 |",
            "| DETR | 0.500s | 1 | 0 | 1 |",
        ]

    def test_missing_model_gets_na_row(self):
        """A model absent from results is shown as N/A rather than omitted."""
        from src.ui.gradio_app import CCTVDetectionApp

        text = CCTVDetectionApp._format_comparison_metrics({"YOLOv8": (None, [], 0.1)})

        assert "| Faster R-CNN | N/A | N/A | N/A | N/A |" in text
        assert "| DETR | N/A | N/A | N/A | N/A |" in text


class TestDetectSingle:
    """detect_single guards its inputs and delegates to the chosen detector."""

    def test_returns_prompt_when_image_missing(self):
        """No image yields no output image and an upload prompt."""
        app = _app_with_detectors({"YOLOv8": MagicMock()})

        image, text = app.detect_single(None, "YOLOv8", 0.5)

        assert image is None
        assert text == "Please upload an image"

    def test_reports_unloaded_model(self):
        """Selecting a model that failed to load returns an explanatory message."""
        app = _app_with_detectors({"YOLOv8": MagicMock()})

        image, text = app.detect_single(Image.new("RGB", (8, 8)), "DETR", 0.5)

        assert image is None
        assert text == "Model DETR not loaded"

    def test_runs_detector_and_formats_metrics(self):
        """The detector's annotated image and detections are returned."""
        annotated = Image.new("RGB", (8, 8))
        detector = MagicMock()
        detector.annotate_image.return_value = annotated
        detector.predict.return_value = [_detection("CCTV", 0.9)]
        app = _app_with_detectors({"YOLOv8": detector})
        source = Image.new("RGB", (8, 8))

        image, text = app.detect_single(source, "YOLOv8", 0.3)

        assert image is annotated
        detector.annotate_image.assert_called_once_with(source, 0.3)
        detector.predict.assert_called_once_with(source, 0.3)
        assert "| **Total Detections** | 1 |" in text
        assert "| **CCTV Cameras** | 1 |" in text


class TestDetectComparison:
    """detect_comparison runs every loaded detector and orders the outputs."""

    def test_returns_prompt_when_image_missing(self):
        """No image yields three empty outputs and an upload prompt."""
        app = _app_with_detectors({"YOLOv8": MagicMock()})

        assert app.detect_comparison(None, 0.5) == (
            None,
            None,
            None,
            "Please upload an image",
        )

    def test_orders_images_by_model_and_handles_missing_models(self):
        """Outputs come back as (YOLO, Faster R-CNN, DETR); missing models are None."""
        yolo_img, detr_img = Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))
        yolo = MagicMock()
        yolo.annotate_image.return_value = yolo_img
        yolo.predict.return_value = [_detection("CCTV", 0.9)]
        detr = MagicMock()
        detr.annotate_image.return_value = detr_img
        detr.predict.return_value = []
        app = _app_with_detectors({"DETR": detr, "YOLOv8": yolo})

        out_yolo, out_frcnn, out_detr, text = app.detect_comparison(
            Image.new("RGB", (4, 4)), 0.5
        )

        assert out_yolo is yolo_img
        assert out_frcnn is None
        assert out_detr is detr_img
        assert "| Faster R-CNN | N/A | N/A | N/A | N/A |" in text
        assert "| YOLOv8 |" in text and "| 1 | 1 | 0 |" in text


class TestLoadDetector:
    """_load_detector applies the Faster R-CNN device override and swallows errors."""

    @patch("src.ui.gradio_app.DetectorFactory.create_detector")
    def test_faster_rcnn_falls_back_to_cpu_on_mps(self, mock_create):
        """On an MPS device, Faster R-CNN is created on CPU instead."""
        app = _app_with_detectors({})
        app.device = torch.device("mps")

        detector = app._load_detector("faster-rcnn", Path("weights.pt"))

        assert detector is mock_create.return_value
        assert mock_create.call_args.kwargs["device"] == torch.device("cpu")
        assert mock_create.call_args.kwargs["model_type"] == "faster-rcnn"

    @patch("src.ui.gradio_app.DetectorFactory.create_detector")
    def test_other_models_keep_app_device(self, mock_create):
        """YOLO and DETR are created on the app's selected device."""
        app = _app_with_detectors({})
        app.device = torch.device("mps")

        app._load_detector("detr", Path("weights"))

        assert mock_create.call_args.kwargs["device"] == torch.device("mps")

    @patch("src.ui.gradio_app.DetectorFactory.create_detector")
    def test_returns_none_when_loading_fails(self, mock_create):
        """A failure inside the factory is reported as None, not raised."""
        mock_create.side_effect = RuntimeError("no weights")
        app = _app_with_detectors({})
        app.device = torch.device("cpu")

        assert app._load_detector("yolo", Path("missing.pt")) is None
