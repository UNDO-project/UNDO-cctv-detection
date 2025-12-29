"""Unit tests for gradio_app module.

This module tests the Gradio UI for CCTV detection.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock YOLO class before any imports to avoid loading the actual model file
@pytest.fixture(scope="module", autouse=True)
def mock_yolo_class():
    """Mock YOLO class for the entire test module.

    :return: MagicMock instance
    """
    with patch("ultralytics.YOLO") as mock_yolo:
        # Clear cached module if it exists
        if "src.ui.gradio_app" in sys.modules:
            del sys.modules["src.ui.gradio_app"]

        # Configure the mock to return a MagicMock instance
        mock_yolo.return_value = MagicMock()

        # Now import the module with the mock in place
        import src.ui.gradio_app  # noqa: F401

        yield mock_yolo


class TestCreateDemo:
    """Test create_demo function."""

    @patch("src.ui.gradio_app.YOLO")
    def test_create_demo_returns_gradio_blocks(self, mock_yolo_class):
        """Test that create_demo returns a Gradio Blocks object.

        :param mock_yolo_class: Mocked YOLO class
        :return: None
        """
        from src.ui.gradio_app import create_demo

        # Mock the model instance
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        demo = create_demo()

        # Test that demo was created successfully
        assert demo is not None
        # Gradio Blocks objects have a launch method
        assert hasattr(demo, "launch")

    @patch("src.ui.gradio_app.YOLO")
    def test_create_demo_loads_model(self, mock_yolo_class):
        """Test that create_demo loads the YOLO model.

        :param mock_yolo_class: Mocked YOLO class
        :return: None
        """
        from src.ui.gradio_app import create_demo

        # Mock the model instance
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        create_demo()

        # Verify YOLO was instantiated
        mock_yolo_class.assert_called_once()


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
