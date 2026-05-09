"""Remove incomplete training run directories under ``runs/``.

A run is considered incomplete when its expected success-marker file is
missing:

- YOLO          : ``runs/detect/train*/results.csv``
- Faster R-CNN  : ``runs/faster_rcnn/train*/training_metrics.json``
- DETR          : ``runs/detr/checkpoint-*/trainer_state.json``

This script complements the trainer-level cleanup-on-abort logic in
``YoloUltralyticsTrainer``: it covers cases where the trainer process was
killed too hard for Python's ``finally`` blocks to run (``kill -9``,
OS shutdown, IDE crash) and provides a one-shot for retroactive cleanup
of historical leftovers.
"""

import argparse
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from src.config import settings

# (subdirectory, glob pattern, expected marker file relative to the run dir)
_RUN_GROUPS: list[tuple[str, str, str]] = [
    ("runs/detect", "train*", "results.csv"),
    ("runs/faster_rcnn", "train*", "training_metrics.json"),
    ("runs/detr", "checkpoint-*", "trainer_state.json"),
]


def _find_incomplete_runs(project_root: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(run_dir, marker_filename)`` for every incomplete run.

    :param project_root: Repository root (parent of ``runs/``)
    :return: Iterator of incomplete run directories with their expected marker
    :rtype: Iterator[tuple[Path, str]]
    """
    for subdir, pattern, marker in _RUN_GROUPS:
        group_root = project_root / subdir
        if not group_root.exists():
            continue
        for run_dir in group_root.glob(pattern):
            if run_dir.is_dir() and not (run_dir / marker).exists():
                yield run_dir, marker


def main() -> None:
    """Entry point for ``cctv-clean-runs``.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description=(
            "Remove training run directories that never produced their "
            "expected success-marker file."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually delete. Without this flag, only show what would be removed.",
    )
    args = parser.parse_args()

    project_root = settings.paths.project_root
    incomplete = list(_find_incomplete_runs(project_root))

    if not incomplete:
        logger.info("No incomplete training runs found.")
        return

    action = "Deleting" if args.force else "Would delete"
    for run_dir, marker in incomplete:
        rel = run_dir.relative_to(project_root)
        logger.info(f"{action}: {rel} (no {marker})")
        if args.force:
            shutil.rmtree(run_dir, ignore_errors=True)

    logger.info(
        f"{'Removed' if args.force else 'Found'} {len(incomplete)} incomplete run(s)."
    )
    if not args.force:
        logger.info("Re-run with --force to actually delete.")


if __name__ == "__main__":
    sys.exit(main())
