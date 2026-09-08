"""Unit tests for MetricsAggregator.

Exercises the per-model loaders against files laid out the way each
training framework writes them, plus the run discovery and latest-run
selection logic, all on a temporary ``runs/`` tree.
"""

import json
from pathlib import Path

import pytest

from src.application.metrics_aggregator import MetricsAggregator


def _write_yolo_results(run_dir: Path, epochs: int, map50: float = 0.8) -> None:
    """Write a results.csv the way Ultralytics does (space-padded headers)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "                  epoch,      train/box_loss,        val/box_loss,"
        "       metrics/mAP50(B),  metrics/mAP50-95(B)\n"
    )
    rows = [
        f"{i},{1.0 - i * 0.1:.3f},{1.2 - i * 0.1:.3f},{map50 - 0.1 + i * 0.01:.3f},0.5\n"
        for i in range(epochs)
    ]
    (run_dir / "results.csv").write_text(header + "".join(rows))


def _write_detr_checkpoint(
    checkpoint_dir: Path, epoch: int, log_history: list[dict]
) -> None:
    """Write a HuggingFace trainer_state.json into a checkpoint directory."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps({"epoch": epoch, "log_history": log_history})
    )


class TestLoadYoloMetrics:
    """load_yolo_metrics reads Ultralytics results.csv."""

    def test_returns_empty_dict_when_results_missing(self, tmp_path: Path):
        """A run directory without results.csv is reported as empty."""
        assert MetricsAggregator.load_yolo_metrics(tmp_path) == {}

    def test_reads_final_row_and_strips_column_whitespace(self, tmp_path: Path):
        """Final mAP comes from the last row; padded headers are handled."""
        _write_yolo_results(tmp_path, epochs=3, map50=0.8)

        metrics = MetricsAggregator.load_yolo_metrics(tmp_path)

        assert metrics["model"] == "YOLOv8"
        assert metrics["epochs"] == 3
        assert metrics["final_map50"] == pytest.approx(0.72)
        assert metrics["final_map"] == pytest.approx(0.5)
        assert metrics["train_loss"] == pytest.approx([1.0, 0.9, 0.8])
        assert metrics["val_loss"] == pytest.approx([1.2, 1.1, 1.0])

    def test_returns_empty_dict_on_malformed_csv(self, tmp_path: Path):
        """A results.csv missing the expected columns is skipped, not raised."""
        (tmp_path / "results.csv").write_text("epoch,foo\n1,2\n")
        assert MetricsAggregator.load_yolo_metrics(tmp_path) == {}


class TestLoadDetrMetrics:
    """load_detr_metrics reads trainer_state.json and optional eval metrics."""

    def test_returns_empty_dict_when_state_missing(self, tmp_path: Path):
        """A checkpoint without trainer_state.json is reported as empty."""
        assert MetricsAggregator.load_detr_metrics(tmp_path) == {}

    def test_splits_train_and_eval_loss_entries(self, tmp_path: Path):
        """Train losses and eval losses are separated with their epochs."""
        checkpoint = tmp_path / "detr" / "checkpoint-20"
        _write_detr_checkpoint(
            checkpoint,
            epoch=2,
            log_history=[
                {"loss": 3.0, "epoch": 0.5},
                {"loss": 2.5, "epoch": 1.0},
                {"eval_loss": 2.8, "epoch": 1.0},
                {"loss": 2.0, "epoch": 2.0},
                {"eval_loss": 2.2, "epoch": 2.0},
                {"train_runtime": 12.0, "epoch": 2.0},
            ],
        )

        metrics = MetricsAggregator.load_detr_metrics(checkpoint)

        assert metrics["model"] == "DETR"
        assert metrics["epochs"] == 2
        assert metrics["train_loss"] == [3.0, 2.5, 2.0]
        assert metrics["train_epochs"] == [0.5, 1.0, 2.0]
        assert metrics["val_loss"] == [2.8, 2.2]
        assert metrics["val_epochs"] == [1.0, 2.0]

    def test_map_defaults_to_zero_without_evaluation_file(self, tmp_path: Path):
        """No final/evaluation_metrics.json means mAP is reported as 0."""
        checkpoint = tmp_path / "detr" / "checkpoint-20"
        _write_detr_checkpoint(checkpoint, epoch=1, log_history=[])

        metrics = MetricsAggregator.load_detr_metrics(checkpoint)

        assert metrics["final_map50"] == 0.0
        assert metrics["final_map"] == 0.0

    def test_reads_map_from_sibling_final_directory(self, tmp_path: Path):
        """mAP is read from runs/detr/final/evaluation_metrics.json."""
        checkpoint = tmp_path / "detr" / "checkpoint-20"
        _write_detr_checkpoint(checkpoint, epoch=1, log_history=[])
        final_dir = tmp_path / "detr" / "final"
        final_dir.mkdir()
        (final_dir / "evaluation_metrics.json").write_text(
            json.dumps({"map50": 0.61, "map": 0.42})
        )

        metrics = MetricsAggregator.load_detr_metrics(checkpoint)

        assert metrics["final_map50"] == pytest.approx(0.61)
        assert metrics["final_map"] == pytest.approx(0.42)

    def test_returns_empty_dict_on_invalid_json(self, tmp_path: Path):
        """A corrupt trainer_state.json is skipped, not raised."""
        (tmp_path / "trainer_state.json").write_text("{not json")
        assert MetricsAggregator.load_detr_metrics(tmp_path) == {}


class TestLoadFasterRcnnMetrics:
    """load_faster_rcnn_metrics reads training_metrics.json."""

    def test_returns_empty_dict_when_file_missing(self, tmp_path: Path):
        """A run directory without training_metrics.json is reported as empty."""
        assert MetricsAggregator.load_faster_rcnn_metrics(tmp_path) == {}

    def test_maps_json_keys_to_metrics(self, tmp_path: Path):
        """train_losses/val_losses keys are renamed to train_loss/val_loss."""
        (tmp_path / "training_metrics.json").write_text(
            json.dumps(
                {
                    "final_map50": 0.7,
                    "final_map": 0.45,
                    "train_losses": [1.0, 0.8],
                    "val_losses": [1.1, 0.9],
                    "epochs": 2,
                }
            )
        )

        metrics = MetricsAggregator.load_faster_rcnn_metrics(tmp_path)

        assert metrics["model"] == "Faster R-CNN"
        assert metrics["final_map50"] == pytest.approx(0.7)
        assert metrics["train_loss"] == [1.0, 0.8]
        assert metrics["val_loss"] == [1.1, 0.9]
        assert metrics["epochs"] == 2

    def test_missing_keys_fall_back_to_defaults(self, tmp_path: Path):
        """Absent keys yield zero mAP, empty losses and zero epochs."""
        (tmp_path / "training_metrics.json").write_text("{}")

        metrics = MetricsAggregator.load_faster_rcnn_metrics(tmp_path)

        assert metrics["final_map50"] == 0.0
        assert metrics["train_loss"] == []
        assert metrics["epochs"] == 0


class TestGetAllMetrics:
    """get_all_metrics discovers runs under the runs directory."""

    def test_returns_empty_list_when_runs_dir_missing(self, tmp_path: Path):
        """A non-existent runs directory yields no metrics."""
        aggregator = MetricsAggregator(tmp_path / "does_not_exist")
        assert aggregator.get_all_metrics() == []

    def test_collects_one_entry_per_model_family(self, tmp_path: Path):
        """YOLO, DETR and Faster R-CNN runs are all discovered."""
        _write_yolo_results(tmp_path / "detect" / "train", epochs=2)
        _write_detr_checkpoint(
            tmp_path / "detr" / "checkpoint-5", epoch=1, log_history=[]
        )
        frcnn = tmp_path / "faster_rcnn" / "train"
        frcnn.mkdir(parents=True)
        (frcnn / "training_metrics.json").write_text(json.dumps({"epochs": 1}))

        models = {m["model"] for m in MetricsAggregator(tmp_path).get_all_metrics()}

        assert models == {"YOLOv8", "DETR", "Faster R-CNN"}

    def test_skips_incomplete_runs(self, tmp_path: Path):
        """Run directories without result files are ignored."""
        _write_yolo_results(tmp_path / "detect" / "train", epochs=2)
        (tmp_path / "detect" / "train2").mkdir()
        (tmp_path / "faster_rcnn" / "train").mkdir(parents=True)

        metrics = MetricsAggregator(tmp_path).get_all_metrics()

        assert len(metrics) == 1
        assert metrics[0]["model"] == "YOLOv8"

    def test_uses_highest_numbered_detr_checkpoint(self, tmp_path: Path):
        """Checkpoints are ordered numerically, so checkpoint-10 beats checkpoint-9."""
        _write_detr_checkpoint(
            tmp_path / "detr" / "checkpoint-9", epoch=1, log_history=[]
        )
        _write_detr_checkpoint(
            tmp_path / "detr" / "checkpoint-10", epoch=2, log_history=[]
        )

        metrics = MetricsAggregator(tmp_path).get_all_metrics()

        assert len(metrics) == 1
        assert metrics[0]["run_dir"].endswith("checkpoint-10")
        assert metrics[0]["epochs"] == 2


class TestGetLatestMetricsPerModel:
    """get_latest_metrics_per_model keeps the longest run per model."""

    def test_keeps_run_with_most_epochs(self, tmp_path: Path):
        """Of two YOLO runs, the one with more epochs is selected."""
        _write_yolo_results(tmp_path / "detect" / "train", epochs=5)
        _write_yolo_results(tmp_path / "detect" / "train2", epochs=20)

        latest = MetricsAggregator(tmp_path).get_latest_metrics_per_model()

        assert set(latest) == {"YOLOv8"}
        assert latest["YOLOv8"]["epochs"] == 20
        assert latest["YOLOv8"]["run_dir"].endswith("train2")

    def test_returns_empty_dict_when_no_runs(self, tmp_path: Path):
        """An empty runs directory yields an empty mapping."""
        assert MetricsAggregator(tmp_path).get_latest_metrics_per_model() == {}
