"""Unit tests for the DatasetPreparation application service.

The service is a one-line delegation to the image converter, so a single
wiring test is all it needs.
"""

from pathlib import Path
from unittest.mock import Mock

from src.application.dataset_preparation import DatasetPreparation


class TestDatasetPreparation:
    """Test DatasetPreparation application service."""

    def test_prepare_dataset_delegates_to_converter(self):
        """prepare_dataset forwards both folders to the image converter."""
        converter = Mock()
        service = DatasetPreparation(converter)

        service.prepare_dataset(Path("/input"), Path("/output"))

        converter.convert_heic_to_jpg.assert_called_once_with(
            Path("/input"), Path("/output")
        )
