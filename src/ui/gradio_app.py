"""Gradio interface for CCTV detection."""

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.config import settings


def create_demo() -> gr.Blocks:
    """Create and configure Gradio demo for CCTV detection.

    :return: Configured Gradio Blocks demo
    :rtype: gr.Blocks
    """
    model = YOLO(str(settings.models.yolo_weights))

    def detect_objects(image: Image.Image) -> Image.Image:
        """Run YOLOv8 detection on the input image.

        :param image: Input image uploaded by the user
        :return: Image with detected bounding boxes and labels drawn
        :rtype: Image.Image
        """
        img_array = np.array(image)
        results = model.predict(source=img_array, conf=0.25, imgsz=640)
        annotated_img = results[0].plot()
        return Image.fromarray(annotated_img)

    with gr.Blocks(title="CCTV Detection") as demo:
        gr.Markdown("# CCTV Detection with YOLOv8")
        gr.Markdown("Upload an image to detect CCTV cameras and signage.")

        with gr.Row():
            input_img = gr.Image(type="pil", label="Upload Image")
            output_img = gr.Image(type="pil", label="Detections")

        btn = gr.Button("Detect CCTV")
        btn.click(fn=detect_objects, inputs=input_img, outputs=output_img)

    return demo


def launch_ui() -> None:
    """Launch the Gradio web interface for CCTV detection."""
    demo = create_demo()
    demo.launch()
