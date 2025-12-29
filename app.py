"""Main entry point for CCTV Detection Gradio UI."""

from src.ui.gradio_app import create_demo


def main() -> None:
    """Launch the CCTV detection web interface."""
    demo = create_demo()
    demo.launch()


if __name__ == "__main__":
    main()
