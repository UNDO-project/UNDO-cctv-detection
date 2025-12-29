import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.config import settings

model = YOLO(str(settings.models.yolo_weights))


def detect_objects(image: Image.Image) -> Image.Image:
    """
    Runs YOLOv8 detection on the input image.
    :param image: Input image uploaded by the user
    :return: Image with detected bounding boxes and labels drawn.
    """
    img_array = np.array(image)
    results = model.predict(source=img_array, conf=0.25, imgsz=640)
    annotated_img = results[0].plot()

    return Image.fromarray(annotated_img)


demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=gr.Image(type="pil", label="Detected image"),
    title="Custom YoloV8 CCTV detector",
    description="Upload an image for CCTV detection.",
)


def launch_ui() -> None:
    """Launch the Gradio web interface for CCTV detection."""
    demo.launch()


if __name__ == "__main__":
    launch_ui()
