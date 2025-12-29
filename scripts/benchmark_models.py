"""Benchmark inference speed of all detection models.

This script benchmarks the inference performance of YOLOv8, Faster R-CNN,
and DETR models on the same hardware to provide fair speed comparisons.
"""

import time

import numpy as np
from PIL import Image

from src.config import settings
from src.domain.services.object_detector import ObjectDetector
from src.infrastructure.detector_factory import DetectorFactory
from src.infrastructure.device_selector import DeviceSelector


def benchmark_model(detector: ObjectDetector, num_runs: int = 100) -> dict[str, float]:
    """Benchmark inference speed of a detector.

    :param detector: Object detector instance
    :param num_runs: Number of inference runs for benchmarking
    :return: Dictionary with mean, std, min, max inference times
    :rtype: Dict[str, float]
    """
    # Create dummy image (640x640 RGB)
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    )

    # Warmup runs
    print(f"  Warmup ({10} runs)...", end=" ", flush=True)
    for _ in range(10):
        detector.predict(dummy_img)
    print("done")

    # Benchmark runs
    print(f"  Benchmarking ({num_runs} runs)...", end=" ", flush=True)
    times = []
    for _ in range(num_runs):
        start = time.time()
        detector.predict(dummy_img)
        times.append(time.time() - start)
    print("done")

    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def main() -> None:
    """Run benchmarks for all available models.

    :return: None
    """
    device = DeviceSelector.get_optimal_device()
    print(f"Benchmarking on device: {device}\n")

    models = {
        "YOLOv8": ("yolo", settings.models.yolo_weights),
        "Faster R-CNN": ("faster-rcnn", settings.models.faster_rcnn_weights),
        "DETR": ("detr", settings.models.detr_weights),
    }

    results = {}

    for name, (model_type, path) in models.items():
        print(f"Benchmarking {name}...")

        if not path.exists():
            print(f"  ⚠️  Model weights not found at {path}, skipping\n")
            continue

        try:
            # Load detector
            detector = DetectorFactory.create_detector(
                model_type=model_type,  # type: ignore[arg-type]
                model_path=path,
                class_names=["CCTV", "CCTV-SIGNS"],
                device=device,
            )

            # Run benchmark
            stats = benchmark_model(detector, num_runs=100)
            results[name] = stats

            # Print results
            print(
                f"  ✅ Mean: {stats['mean'] * 1000:.1f}ms ± {stats['std'] * 1000:.1f}ms"
            )
            print(
                f"     Range: [{stats['min'] * 1000:.1f}ms, {stats['max'] * 1000:.1f}ms]\n"
            )

        except Exception as e:
            print(f"  ❌ Error benchmarking {name}: {e}\n")

    # Print summary comparison
    if results:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Sort by mean inference time
        sorted_results = sorted(results.items(), key=lambda x: x[1]["mean"])

        print(f"{'Model':<20} {'Mean (ms)':<15} {'Std (ms)':<15}")
        print("-" * 60)

        for model_name, stats in sorted_results:
            print(
                f"{model_name:<20} {stats['mean'] * 1000:>10.2f}     {stats['std'] * 1000:>10.2f}"
            )

        # Show speedup relative to slowest
        slowest_time = sorted_results[-1][1]["mean"]
        print("\n" + "=" * 60)
        print("SPEEDUP (relative to slowest)")
        print("=" * 60)

        for model_name, stats in sorted_results:
            speedup = slowest_time / stats["mean"]
            print(f"{model_name:<20} {speedup:>6.2f}x")

    else:
        print("No models could be benchmarked. Ensure model weights are available.")


if __name__ == "__main__":
    main()
