"""Tests for the ``cctv-clean-runs`` CLI."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.clean_runs import _find_incomplete_runs, main


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    """Build a project tree with a mix of complete and incomplete runs.

    :param tmp_path: pytest tmp dir
    :return: project root
    :rtype: Path
    """
    # YOLO: one complete, one incomplete
    (tmp_path / "runs" / "detect" / "train_ok" / "weights").mkdir(parents=True)
    (tmp_path / "runs" / "detect" / "train_ok" / "results.csv").write_text("x")
    (tmp_path / "runs" / "detect" / "train_bad" / "weights").mkdir(parents=True)

    # Faster R-CNN: one complete
    (tmp_path / "runs" / "faster_rcnn" / "train").mkdir(parents=True)
    (tmp_path / "runs" / "faster_rcnn" / "train" / "training_metrics.json").write_text(
        "{}"
    )

    # DETR: one complete checkpoint, one incomplete
    (tmp_path / "runs" / "detr" / "checkpoint-100").mkdir(parents=True)
    (tmp_path / "runs" / "detr" / "checkpoint-100" / "trainer_state.json").write_text(
        "{}"
    )
    (tmp_path / "runs" / "detr" / "checkpoint-200").mkdir(parents=True)
    return tmp_path


def test_find_incomplete_runs_identifies_missing_markers(fake_project: Path):
    """Only directories without their expected marker file should be flagged."""
    incomplete = {
        run.relative_to(fake_project).as_posix(): marker
        for run, marker in _find_incomplete_runs(fake_project)
    }

    assert incomplete == {
        "runs/detect/train_bad": "results.csv",
        "runs/detr/checkpoint-200": "trainer_state.json",
    }


def test_find_incomplete_runs_handles_missing_subdirs(tmp_path: Path):
    """A project with no runs/ at all should yield nothing without erroring."""
    assert list(_find_incomplete_runs(tmp_path)) == []


def test_main_dry_run_does_not_delete(fake_project: Path):
    """Without --force the CLI must leave the filesystem untouched."""
    fake_settings = SimpleNamespace(paths=SimpleNamespace(project_root=fake_project))
    with (
        patch("scripts.clean_runs.settings", fake_settings),
        patch("sys.argv", ["cctv-clean-runs"]),
    ):
        main()

    assert (fake_project / "runs" / "detect" / "train_bad").exists()
    assert (fake_project / "runs" / "detr" / "checkpoint-200").exists()


def test_main_force_deletes_only_incomplete_runs(fake_project: Path):
    """With --force, exactly the incomplete dirs must be removed."""
    fake_settings = SimpleNamespace(paths=SimpleNamespace(project_root=fake_project))
    with (
        patch("scripts.clean_runs.settings", fake_settings),
        patch("sys.argv", ["cctv-clean-runs", "--force"]),
    ):
        main()

    # Incomplete: gone.
    assert not (fake_project / "runs" / "detect" / "train_bad").exists()
    assert not (fake_project / "runs" / "detr" / "checkpoint-200").exists()
    # Complete: untouched.
    assert (fake_project / "runs" / "detect" / "train_ok" / "results.csv").exists()
    assert (fake_project / "runs" / "faster_rcnn" / "train").exists()
    assert (fake_project / "runs" / "detr" / "checkpoint-100").exists()
