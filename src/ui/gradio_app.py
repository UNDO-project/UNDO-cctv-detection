"""Multi-model CCTV detection Gradio interface."""

import time
from pathlib import Path

import gradio as gr
from PIL import Image

from src.config import settings
from src.infrastructure.detector_factory import DetectorFactory, ModelType
from src.infrastructure.device_selector import DeviceSelector


class CCTVDetectionApp:
    """Gradio app for CCTV detection with multiple models."""

    def __init__(self):
        """Initialize app with all models."""
        self.device = DeviceSelector.get_optimal_device()
        self.class_names = ["CCTV", "CCTV-SIGNS"]

        # Load all models with error handling
        self.detectors = {}
        model_configs = [
            ("YOLOv8", "yolo", settings.models.yolo_weights),
            ("Faster R-CNN", "faster-rcnn", settings.models.faster_rcnn_weights),
            ("DETR", "detr", settings.models.detr_weights),
        ]

        for name, model_type, path in model_configs:
            detector = self._load_detector(model_type, path)  # type: ignore[arg-type]
            if detector:
                self.detectors[name] = detector
                print(f"✅ Loaded {name} model")
            else:
                print(f"⚠️  {name} model not available (missing weights at {path})")

        if not self.detectors:
            raise RuntimeError("No models could be loaded! Ensure model weights exist.")

    def _load_detector(self, model_type: ModelType, model_path: Path):
        """Load detector with error handling.

        :param model_type: Type of model to load
        :param model_path: Path to model weights
        :return: Detector instance or None if loading failed
        """
        try:
            return DetectorFactory.create_detector(
                model_type=model_type,
                model_path=model_path,
                class_names=self.class_names,
                device=self.device,
            )
        except Exception as e:
            print(f"Warning: Could not load {model_type} model: {e}")
            return None

    def detect_single(
        self,
        image: Image.Image,
        model_name: str,
        confidence: float,
    ) -> tuple[Image.Image | None, str]:
        """Run detection with single model.

        :param image: Input image
        :param model_name: Name of model to use
        :param confidence: Confidence threshold
        :return: Tuple of (annotated_image, metrics_text)
        :rtype: tuple[Image.Image | None, str]
        """
        if image is None:
            return None, "Please upload an image"

        detector = self.detectors.get(model_name)
        if detector is None:
            return None, f"Model {model_name} not loaded"

        # Run inference
        start_time = time.time()
        annotated_img = detector.annotate_image(image, confidence)
        detections = detector.predict(image, confidence)
        inference_time = time.time() - start_time

        # Format metrics
        metrics = self._format_metrics(model_name, detections, inference_time)

        return annotated_img, metrics

    def detect_comparison(
        self,
        image: Image.Image,
        confidence: float,
    ) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, str]:
        """Run detection with all three models for comparison.

        :param image: Input image
        :param confidence: Confidence threshold
        :return: Tuple of (yolo_img, faster_rcnn_img, detr_img, comparison_metrics)
        :rtype: tuple[Image.Image | None, Image.Image | None, Image.Image | None, str]
        """
        if image is None:
            return None, None, None, "Please upload an image"

        results = {}
        for model_name, detector in self.detectors.items():
            if detector is None:
                results[model_name] = (None, [], 0)
                continue

            start_time = time.time()
            annotated_img = detector.annotate_image(image, confidence)
            detections = detector.predict(image, confidence)
            inference_time = time.time() - start_time

            results[model_name] = (annotated_img, detections, inference_time)

        # Format comparison metrics
        comparison_text = self._format_comparison_metrics(results)

        return (
            results.get("YOLOv8", (None, [], 0))[0],
            results.get("Faster R-CNN", (None, [], 0))[0],
            results.get("DETR", (None, [], 0))[0],
            comparison_text,
        )

    @staticmethod
    def _format_metrics(
        model_name: str,
        detections: list,
        inference_time: float,
    ) -> str:
        """Format metrics as Markdown table.

        :param model_name: Name of the model
        :param detections: List of detections
        :param inference_time: Inference time in seconds
        :return: Formatted Markdown metrics
        :rtype: str
        """
        cctv_count = sum(1 for d in detections if d["class_name"] == "CCTV")
        sign_count = sum(1 for d in detections if d["class_name"] == "CCTV-SIGNS")

        metrics = f"""
### {model_name} Results

| Metric | Value |
|--------|-------|
| **Inference Time** | {inference_time:.3f}s |
| **Total Detections** | {len(detections)} |
| **CCTV Cameras** | {cctv_count} |
| **CCTV Signs** | {sign_count} |

#### Detections:
"""
        for i, det in enumerate(detections, 1):
            metrics += (
                f"\n{i}. **{det['class_name']}** - Confidence: {det['confidence']:.2%}"
            )

        return metrics

    @staticmethod
    def _format_comparison_metrics(results: dict) -> str:
        """Format comparison metrics for all models.

        :param results: Dictionary of model results
        :return: Formatted Markdown comparison table
        :rtype: str
        """
        comparison = """
### Model Comparison

| Model | Inference Time | Detections | CCTV | Signs |
|-------|---------------|------------|------|-------|
"""
        for model_name in ["YOLOv8", "Faster R-CNN", "DETR"]:
            result = results.get(model_name)
            if result is None or result[1] is None:
                comparison += f"| {model_name} | N/A | N/A | N/A | N/A |\n"
                continue

            _, detections, inf_time = result
            cctv = sum(1 for d in detections if d["class_name"] == "CCTV")
            signs = sum(1 for d in detections if d["class_name"] == "CCTV-SIGNS")
            comparison += f"| {model_name} | {inf_time:.3f}s | {len(detections)} | {cctv} | {signs} |\n"

        return comparison


def create_demo() -> gr.Blocks:
    """Create and configure Gradio demo interface.

    :return: Configured Gradio Blocks interface
    :rtype: gr.Blocks
    """
    app = CCTVDetectionApp()

    # Define theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    )

    with gr.Blocks(
        theme=theme,
        title="CCTV Detection - Multi-Model Comparison",
    ) as demo:
        gr.Markdown(
            """
            # 🎥 CCTV Detection System

            Compare three state-of-the-art object detection models:
            - **YOLOv8**: Fast one-stage CNN detector
            - **Faster R-CNN**: Accurate two-stage CNN detector
            - **DETR**: Modern transformer-based detector
            """
        )

        with gr.Tabs():
            # ==================== Single Model Tab ====================
            with gr.Tab("Single Model Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_input = gr.Image(
                            type="pil",
                            label="Upload Image",
                            height=400,
                        )
                        model_selector = gr.Dropdown(
                            choices=list(app.detectors.keys()),
                            value=list(app.detectors.keys())[0]
                            if app.detectors
                            else None,
                            label="Select Model",
                        )
                        confidence_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            value=0.25,
                            step=0.05,
                            label="Confidence Threshold",
                        )
                        detect_btn = gr.Button(
                            "🔍 Detect CCTV",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        single_output = gr.Image(
                            type="pil",
                            label="Detections",
                            height=400,
                        )
                        single_metrics = gr.Markdown(
                            "Upload an image and click Detect to see results"
                        )

                detect_btn.click(
                    fn=app.detect_single,
                    inputs=[single_input, model_selector, confidence_slider],
                    outputs=[single_output, single_metrics],
                )

            # ==================== Comparison Tab ====================
            with gr.Tab("Model Comparison"):
                gr.Markdown(
                    """
                    ### Compare All Models Side-by-Side
                    Upload an image to see how each model performs on the same input.
                    """
                )

                with gr.Row():
                    comparison_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        height=300,
                    )

                comparison_confidence = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold (all models)",
                )

                compare_btn = gr.Button(
                    "🔍 Compare Models",
                    variant="primary",
                    size="lg",
                )

                with gr.Row():
                    yolo_output = gr.Image(
                        type="pil",
                        label="YOLOv8",
                        height=300,
                    )
                    faster_rcnn_output = gr.Image(
                        type="pil",
                        label="Faster R-CNN",
                        height=300,
                    )
                    detr_output = gr.Image(
                        type="pil",
                        label="DETR",
                        height=300,
                    )

                comparison_metrics = gr.Markdown(
                    "Click 'Compare Models' to see results"
                )

                compare_btn.click(
                    fn=app.detect_comparison,
                    inputs=[comparison_input, comparison_confidence],
                    outputs=[
                        yolo_output,
                        faster_rcnn_output,
                        detr_output,
                        comparison_metrics,
                    ],
                )

            # ==================== Examples Tab ====================
            with gr.Tab("Example Images"):
                gr.Markdown(
                    """
                    ### Example CCTV Images
                    Click any example to load it in the Single Model tab.
                    """
                )

                # Find example images
                examples_dir = settings.paths.project_root / "examples"
                if examples_dir.exists():
                    example_images = sorted(examples_dir.glob("*.jpg"))[:6]
                else:
                    example_images = []

                if example_images:
                    gr.Examples(
                        examples=[[str(img)] for img in example_images],
                        inputs=single_input,
                        label="Click to load example",
                    )
                else:
                    gr.Markdown(
                        "⚠️ No example images found. "
                        "Add images to `examples/` directory or run `uv run python scripts/prepare_examples.py`."
                    )

            # ==================== About Tab ====================
            with gr.Tab("About"):
                gr.Markdown(
                    f"""
                    ### CCTV Detection System

                    **Version:** 0.2.0
                    **Device:** {app.device}

                    #### Model Details

                    **YOLOv8 (You Only Look Once v8)**
                    - Architecture: One-stage CNN detector
                    - Speed: ⚡ Fastest (~10-50ms per image)
                    - Use case: Real-time applications, edge deployment
                    - Trained on: Custom CCTV dataset (20 epochs)

                    **Faster R-CNN (Region-based CNN)**
                    - Architecture: Two-stage CNN detector
                    - Speed: 🐢 Slower (~100-300ms per image)
                    - Use case: Accuracy-critical applications
                    - Backbone: ResNet-50 with Feature Pyramid Network

                    **DETR (DEtection TRansformer)**
                    - Architecture: Transformer-based end-to-end detector
                    - Speed: 🐌 Slowest (~200-500ms per image)
                    - Use case: Research, no NMS required
                    - Backbone: ResNet-50 + Transformer encoder-decoder

                    #### Dataset
                    - **Classes:** CCTV (cameras), CCTV-SIGNS (signage)
                    - **Source:** Ethnographic research + public datasets

                    #### Performance Metrics
                    Models trained on the same CCTV dataset:
                    - Training: 70% split
                    - Validation: 30% split
                    - Evaluation metric: mAP@0.5 (IoU threshold 0.5)

                    #### Citation
                    If you use this system in research, please cite:
                    ```
                    @software{{cctv_detection_2024,
                      title={{CCTV Detection System}},
                      author={{UNDO Project}},
                      year={{2024}},
                      url={{https://github.com/jethronap/cctv_detection}}
                    }}
                    ```

                    ---

                    **License:** CC0 1.0 Universal (Public Domain)
                    **Repository:** [GitHub](https://github.com/jethronap/cctv_detection)
                    """
                )

    return demo


def launch_ui() -> None:
    """Launch the Gradio web interface for CCTV detection."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
