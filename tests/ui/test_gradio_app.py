"""Unit tests for gradio_app module.

This module tests the Gradio UI for CCTV detection.
"""

from unittest.mock import MagicMock, patch


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
