"""Aggregate and analyze model training metrics.

This module provides utilities for loading and aggregating training metrics
from different model architectures (YOLO, DETR, Faster R-CNN) for analysis
and visualization.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class MetricsAggregator:
    """Aggregates metrics from training runs for analysis and visualization."""

    def __init__(self, runs_dir: Path) -> None:
        """Initialize with runs directory.

        :param runs_dir: Directory containing training runs
        """
        self.runs_dir = runs_dir

    @staticmethod
    def load_yolo_metrics(run_dir: Path) -> dict[str, Any]:
        """Load metrics from YOLO training run.

        :param run_dir: Directory containing YOLO training results
        :return: Dictionary with model metrics (model, map50, losses, epochs)
        :rtype: dict[str, any]
        """
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            logger.warning(f"No results.csv in {run_dir}")
            return {}

        try:
            df = pd.read_csv(results_csv)

            # Strip whitespace from column names
            df.columns = df.columns.str.strip()

            return {
                "model": "YOLOv8",
                "run_dir": str(run_dir),
                "final_map50": float(df["metrics/mAP50(B)"].iloc[-1]),
                "final_map": float(df["metrics/mAP50-95(B)"].iloc[-1]),
                "train_loss": df["train/box_loss"].tolist(),
                "val_loss": df["val/box_loss"].tolist(),
                "epochs": len(df),
            }
        except Exception as e:
            logger.error(f"Failed to load YOLO metrics from {run_dir}: {e}")
            return {}

    @staticmethod
    def load_detr_metrics(run_dir: Path) -> dict[str, Any]:
        """Load metrics from DETR training run.

        :param run_dir: Directory containing DETR training checkpoint
        :return: Dictionary with model metrics
        :rtype: dict[str, any]
        """
        # Look for trainer_state.json from HuggingFace Trainer
        trainer_state = run_dir / "trainer_state.json"
        if not trainer_state.exists():
            logger.warning(f"No trainer_state.json in {run_dir}")
            return {}

        try:
            import json

            with open(trainer_state) as f:
                state = json.load(f)

            # Extract loss history from log_history
            train_losses = []
            eval_losses = []

            for entry in state.get("log_history", []):
                if "loss" in entry:
                    train_losses.append(entry["loss"])
                if "eval_loss" in entry:
                    eval_losses.append(entry["eval_loss"])

            # Get final metrics
            final_metrics = state.get("best_metric")

            return {
                "model": "DETR",
                "run_dir": str(run_dir),
                "final_map50": final_metrics if final_metrics else 0.0,
                "final_map": 0.0,  # DETR doesn't compute full mAP by default
                "train_loss": train_losses,
                "val_loss": eval_losses,
                "epochs": state.get("epoch", 0),
            }
        except Exception as e:
            logger.error(f"Failed to load DETR metrics from {run_dir}: {e}")
            return {}

    @staticmethod
    def load_faster_rcnn_metrics(run_dir: Path) -> dict[str, Any]:
        """Load metrics from Faster R-CNN training run.

        :param run_dir: Directory containing Faster R-CNN training logs
        :return: Dictionary with model metrics
        :rtype: dict[str, any]
        """
        # Look for training_metrics.json or similar
        metrics_file = run_dir / "training_metrics.json"
        if not metrics_file.exists():
            logger.warning(f"No training_metrics.json in {run_dir}")
            return {}

        try:
            import json

            with open(metrics_file) as f:
                metrics = json.load(f)

            return {
                "model": "Faster R-CNN",
                "run_dir": str(run_dir),
                "final_map50": metrics.get("final_map50", 0.0),
                "final_map": metrics.get("final_map", 0.0),
                "train_loss": metrics.get("train_losses", []),
                "val_loss": metrics.get("val_losses", []),
                "epochs": metrics.get("epochs", 0),
            }
        except Exception as e:
            logger.error(f"Failed to load Faster R-CNN metrics from {run_dir}: {e}")
            return {}

    def get_all_metrics(self) -> list[dict[str, Any]]:
        """Load metrics from all model runs.

        :return: List of metric dictionaries for all available models
        :rtype: list[dict[str, any]]
        """
        all_metrics = []

        if not self.runs_dir.exists():
            logger.warning(f"Runs directory does not exist: {self.runs_dir}")
            return all_metrics

        # Find YOLO runs
        yolo_pattern = "detect/train*"
        for run_dir in self.runs_dir.glob(yolo_pattern):
            if run_dir.is_dir():
                metrics = self.load_yolo_metrics(run_dir)
                if metrics:
                    all_metrics.append(metrics)
                    logger.info(f"Loaded YOLO metrics from {run_dir.name}")

        # Find DETR runs
        detr_pattern = "detr/checkpoint-*"
        for run_dir in self.runs_dir.glob(detr_pattern):
            if run_dir.is_dir():
                metrics = self.load_detr_metrics(run_dir)
                if metrics:
                    all_metrics.append(metrics)
                    logger.info(f"Loaded DETR metrics from {run_dir.name}")

        # Find Faster R-CNN runs
        fasterrcnn_pattern = "faster_rcnn/train*"
        for run_dir in self.runs_dir.glob(fasterrcnn_pattern):
            if run_dir.is_dir():
                metrics = self.load_faster_rcnn_metrics(run_dir)
                if metrics:
                    all_metrics.append(metrics)
                    logger.info(f"Loaded Faster R-CNN metrics from {run_dir.name}")

        if not all_metrics:
            logger.warning("No training metrics found in runs directory")

        return all_metrics

    def get_latest_metrics_per_model(self) -> dict[str, dict[str, Any]]:
        """Get the latest training run for each model type.

        :return: Dictionary mapping model names to their latest metrics
        :rtype: dict[str, dict[str, any]]
        """
        all_metrics = self.get_all_metrics()

        # Group by model and take the latest
        latest_by_model = {}
        for metrics in all_metrics:
            model_name = metrics["model"]
            if (
                model_name not in latest_by_model
                or metrics["epochs"] > latest_by_model[model_name]["epochs"]
            ):
                latest_by_model[model_name] = metrics

        return latest_by_model
