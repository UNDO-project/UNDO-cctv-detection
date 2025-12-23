"""Unit tests for DatasetPreparation application service.

This module tests the DatasetPreparation service which orchestrates
dataset preparation by converting HEIC images to JPG format.
"""

import pytest
from unittest.mock import Mock
from pathlib import Path

from src.application.dataset_preparation import DatasetPreparation


class TestDatasetPreparation:
    """Test DatasetPreparation application service."""

    @pytest.fixture
    def mock_converter(self):
        """Create a mock ImageConverter.

        :return: Mock ImageConverter instance
        """
        return Mock()

    @pytest.fixture
    def service(self, mock_converter):
        """Create DatasetPreparation with mocked converter.

        :param mock_converter: Mocked image converter
        :return: DatasetPreparation instance
        """
        return DatasetPreparation(mock_converter)

    def test_initialization(self, mock_converter):
        """Test that service initializes with image converter.

        :param mock_converter: Mocked image converter
        :return: None
        """
        service = DatasetPreparation(mock_converter)
        assert service.image_converter is mock_converter

    def test_prepare_dateset_delegates_to_converter(self, service, mock_converter):
        """Test that prepare_dateset delegates to image converter.

        :param service: DatasetPreparation instance
        :param mock_converter: Mocked image converter
        :return: None
        """
        input_folder = Path("/input")
        output_folder = Path("/output")

        service.prepare_dataset(input_folder, output_folder)

        mock_converter.convert_heic_to_jpg.assert_called_once_with(
            input_folder, output_folder
        )

    def test_prepare_dateset_with_different_paths(self, service, mock_converter):
        """Test preparation with various folder paths.

        :param service: DatasetPreparation instance
        :param mock_converter: Mocked image converter
        :return: None
        """
        test_cases = [
            (Path("/data/input"), Path("/data/output")),
            (Path("relative/input"), Path("relative/output")),
            (Path("/home/user/images"), Path("/home/user/converted")),
        ]

        for input_path, output_path in test_cases:
            mock_converter.reset_mock()
            service.prepare_dataset(input_path, output_path)

            mock_converter.convert_heic_to_jpg.assert_called_once_with(
                input_path, output_path
            )

    def test_prepare_dateset_returns_none(self, service, mock_converter):
        """Test that prepare_dateset returns None.

        :param service: DatasetPreparation instance
        :param mock_converter: Mocked image converter
        :return: None
        """
        result = service.prepare_dataset(Path("/input"), Path("/output"))
        assert result is None
