"""Multi-model CCTV detection Gradio interface."""

import time
from pathlib import Path

import gradio as gr
from PIL import Image

from src.application.metrics_aggregator import MetricsAggregator
from src.config import settings
from src.domain.services.object_detector import ObjectDetector
from src.infrastructure.detector_factory import DetectorFactory, ModelType
from src.infrastructure.device_selector import DeviceSelector
from src.ui.visualizations import (
    create_inference_benchmark_chart,
    create_loss_curve,
    create_map_comparison,
    create_metrics_summary_table,
)


class CCTVDetectionApp:
    """Gradio app for CCTV detection with multiple models."""

    def __init__(self) -> None:
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

    def _load_detector(
        self, model_type: ModelType, model_path: Path
    ) -> ObjectDetector | None:
        """Load detector with error handling.

        :param model_type: Type of model to load
        :param model_path: Path to model weights
        :return: Detector instance or None if loading failed
        :rtype: ObjectDetector | None
        """
        try:
            import torch

            # Faster R-CNN has MPS compatibility issues
            # Use CUDA if available, otherwise CPU (skip MPS)
            if model_type == "faster-rcnn" and self.device.type == "mps":
                device = torch.device("cpu")
                print("⚠️  Using CPU for Faster R-CNN (MPS not compatible)")
            else:
                device = self.device

            return DetectorFactory.create_detector(
                model_type=model_type,
                model_path=model_path,
                class_names=self.class_names,
                device=device,
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

    with gr.Blocks(title="CCTV Detection - Multi-Model Comparison") as demo:
        gr.Markdown(
            """
            # CCTV Detection System

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

            # ==================== Performance Dashboard Tab ====================
            with gr.Tab("📊 Performance Dashboard"):
                gr.Markdown("### Model Training Metrics")

                # Load metrics
                runs_dir = settings.paths.project_root / "runs"
                aggregator = MetricsAggregator(runs_dir)
                all_metrics = aggregator.get_all_metrics()

                if all_metrics:
                    # Filter to best runs only
                    # For models with mAP: highest mAP@0.5
                    # For models without mAP: most epochs trained
                    best_runs = {}
                    for m in all_metrics:
                        model_name = m["model"]
                        map50_val = m.get("final_map50", 0.0)
                        epochs = m.get("epochs", 0)

                        if model_name not in best_runs:
                            best_runs[model_name] = m
                        else:
                            existing_map = best_runs[model_name].get("final_map50", 0.0)
                            existing_epochs = best_runs[model_name].get("epochs", 0)

                            if map50_val > 0 and existing_map > 0:
                                # Both have mAP, compare mAP
                                if map50_val > existing_map:
                                    best_runs[model_name] = m
                            else:
                                # No mAP available, compare epochs
                                if epochs > existing_epochs:
                                    best_runs[model_name] = m

                    # Sort: models with mAP first (by mAP), then by epochs
                    filtered_metrics = sorted(
                        best_runs.values(),
                        key=lambda x: (x.get("final_map50", 0.0), x.get("epochs", 0)),
                        reverse=True,
                    )

                    # Metrics summary table
                    gr.Markdown("#### Training Summary")
                    summary_table = create_metrics_summary_table(all_metrics)
                    gr.Plot(summary_table)

                    # mAP comparison chart
                    gr.Markdown("#### Model Performance Comparison")
                    map_plot = create_map_comparison(filtered_metrics)
                    gr.Plot(map_plot)

                    # Individual loss curves (best run per model)
                    gr.Markdown("#### Training History (Best Runs)")
                    for metrics in filtered_metrics:
                        loss_plot = create_loss_curve(metrics)
                        gr.Plot(loss_plot)

                    # Inference speed benchmark
                    gr.Markdown("---")
                    gr.Markdown("#### Inference Speed Benchmark")

                    # Load benchmark results if available
                    benchmark_file = (
                        settings.paths.project_root / "runs" / "benchmark_results.json"
                    )
                    if benchmark_file.exists():
                        import json

                        with open(benchmark_file) as f:
                            benchmark_results = json.load(f)

                        benchmark_plot = create_inference_benchmark_chart(
                            benchmark_results
                        )
                        gr.Plot(benchmark_plot)

                        gr.Markdown(
                            """
                            *Results shown above. To re-run benchmarks with latest models:*
                            ```bash
                            uv run cctv-benchmark
                            ```
                            """
                        )
                    else:
                        gr.Markdown(
                            """
                            To benchmark inference speed across all models, run:
                            ```bash
                            uv run cctv-benchmark
                            ```

                            This will measure mean inference time, standard deviation, and speedup comparisons.
                            Results will appear here after benchmarking completes.
                            """
                        )
                else:
                    gr.Markdown(
                        """
                        ⚠️ **No training metrics found.**

                        Train models first to see performance analytics:
                        - **YOLO**: `uv run cctv-train`
                        - **DETR**: `uv run cctv-train-detr`

                        Training metrics will appear here automatically after training completes.
                        """
                    )

            # ==================== About Tab ====================
            with gr.Tab("About"):
                gr.Markdown(
                    f"""
                    ### CCTV Detection System

                    **Version:** 1.0.0
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

                    ##### Metrics Explained

                    **mAP (mean Average Precision)**
                    - Primary metric for object detection quality
                    - Ranges from 0-1 (higher is better)
                    - mAP@0.5: Considers detections correct if IoU ≥ 0.5 with ground truth
                    - mAP@0.5:0.95: Average across IoU thresholds 0.5 to 0.95 (stricter)

                    **IoU (Intersection over Union)**
                    - Measures overlap between predicted and true bounding boxes
                    - IoU = (Area of Overlap) / (Area of Union)
                    - IoU ≥ 0.5 typically considered "correct" detection

                    **Loss Metrics**
                    - Box Loss: Accuracy of bounding box coordinates
                    - Classification Loss: Accuracy of object class predictions
                    - Total Loss: Combined metric used during training (lower is better)

                    **Inference Time**
                    - Time to process a single image (milliseconds)
                    - Measured over 100 runs with 10 warmup iterations
                    - Reported as: mean ± std deviation

                    **Confidence Threshold**
                    - Minimum score for a detection to be shown (0-1 range)
                    - Lower threshold: More detections (may include false positives)
                    - Higher threshold: Fewer, more confident detections

                    #### Citation
                    If you use this system in research, please cite:
                    ```
                    @software{{cctv_detection_2024,
                      title={{CCTV Detection System}},
                      author={{UNDO Project}},
                      year={{2024}},
                      url={{https://github.com/UNDO-project/UNDO-cctv-detection}}
                    }}
                    ```

                    ---

                    **License:** GNU General Public License v3.0
                    **Repository:** [GitHub](https://github.com/UNDO-project/UNDO-cctv-detection)
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
