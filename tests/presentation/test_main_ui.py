"""Unit tests for main_ui module.

This module tests the Gradio UI detection function for YOLOv8 inference.
"""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import sys


# Mock YOLO before importing main_ui to avoid loading actual model
with patch("ultralytics.YOLO"):
    if "src.presentation.main_ui" in sys.modules:
        del sys.modules["src.presentation.main_ui"]


class TestDetectObjects:
    """Test detect_objects function."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL Image.

        :return: PIL Image object
        """
        img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        return Image.fromarray(img_array)

    @patch("src.presentation.main_ui.model")
    def test_detect_objects_returns_image(self, mock_model, sample_image):
        """Test that detect_objects returns a PIL Image.

        :param mock_model: Mocked YOLO model
        :param sample_image: Sample image fixture
        :return: None
        """
        from src.presentation.main_ui import detect_objects

        # Mock the model prediction
        mock_result = MagicMock()
        annotated_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        mock_result.plot.return_value = annotated_array
        mock_model.predict.return_value = [mock_result]

        result = detect_objects(sample_image)

        assert isinstance(result, Image.Image)

    @patch("src.presentation.main_ui.model")
    def test_detect_objects_calls_model_predict(self, mock_model, sample_image):
        """Test that detect_objects calls model.predict with correct parameters.

        :param mock_model: Mocked YOLO model
        :param sample_image: Sample image fixture
        :return: None
        """
        from src.presentation.main_ui import detect_objects

        # Mock the model prediction
        mock_result = MagicMock()
        annotated_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        mock_result.plot.return_value = annotated_array
        mock_model.predict.return_value = [mock_result]

        detect_objects(sample_image)

        # Verify model.predict was called
        mock_model.predict.assert_called_once()
        call_kwargs = mock_model.predict.call_args[1]

        assert call_kwargs["conf"] == 0.25
        assert call_kwargs["imgsz"] == 640

    @patch("src.presentation.main_ui.model")
    def test_detect_objects_uses_correct_confidence(self, mock_model, sample_image):
        """Test that detect_objects uses correct confidence threshold.

        :param mock_model: Mocked YOLO model
        :param sample_image: Sample image fixture
        :return: None
        """
        from src.presentation.main_ui import detect_objects

        # Mock the model prediction
        mock_result = MagicMock()
        annotated_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        mock_result.plot.return_value = annotated_array
        mock_model.predict.return_value = [mock_result]

        detect_objects(sample_image)

        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["conf"] == 0.25

    @patch("src.presentation.main_ui.model")
    def test_detect_objects_converts_image_to_array(self, mock_model, sample_image):
        """Test that detect_objects converts PIL Image to numpy array.

        :param mock_model: Mocked YOLO model
        :param sample_image: Sample image fixture
        :return: None
        """
        from src.presentation.main_ui import detect_objects

        # Mock the model prediction
        mock_result = MagicMock()
        annotated_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        mock_result.plot.return_value = annotated_array
        mock_model.predict.return_value = [mock_result]

        detect_objects(sample_image)

        # Verify predict was called with a numpy array (source parameter)
        call_kwargs = mock_model.predict.call_args[1]
        source = call_kwargs["source"]
        assert isinstance(source, np.ndarray)

    @patch("src.presentation.main_ui.model")
    def test_detect_objects_plots_results(self, mock_model, sample_image):
        """Test that detect_objects calls plot on results.

        :param mock_model: Mocked YOLO model
        :param sample_image: Sample image fixture
        :return: None
        """
        from src.presentation.main_ui import detect_objects

        # Mock the model prediction
        mock_result = MagicMock()
        annotated_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        mock_result.plot.return_value = annotated_array
        mock_model.predict.return_value = [mock_result]

        detect_objects(sample_image)

        # Verify plot was called on the first result
        mock_result.plot.assert_called_once()

    @patch("src.presentation.main_ui.model")
    def test_detect_objects_with_different_image_sizes(self, mock_model):
        """Test detect_objects with various image sizes.

        :param mock_model: Mocked YOLO model
        :return: None
        """
        from src.presentation.main_ui import detect_objects

        # Mock the model prediction
        mock_result = MagicMock()
        annotated_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        mock_result.plot.return_value = annotated_array
        mock_model.predict.return_value = [mock_result]

        # Test with different sizes
        for size in [(320, 320), (640, 480), (1920, 1080)]:
            img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            result = detect_objects(img)

            assert isinstance(result, Image.Image)
            mock_model.predict.assert_called()
