from ultralytics import YOLO
from PIL import Image
import numpy as np
import gradio as gr

model = YOLO("../../samples/best.pt")


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

if __name__ == "__main__":
    demo.launch()
