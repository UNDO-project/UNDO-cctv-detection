"""Launch CCTV Detection Gradio UI.

This script provides the console entry point for the Gradio web interface.
"""

from src.ui.gradio_app import create_demo


def main() -> None:
    """Launch the CCTV detection web interface.

    Creates and launches the Gradio demo with all detection models
    and performance dashboard.

    :return: None
    :rtype: None
    """
    demo = create_demo()
    demo.launch()


if __name__ == "__main__":
    main()
