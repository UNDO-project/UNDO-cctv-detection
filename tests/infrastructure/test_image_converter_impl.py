"""Integration tests for PillowImageConverter.

This module tests the PillowImageConverter which converts HEIC images
to JPEG format using Pillow and pillow_heif.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.infrastructure.image_converter_impl import PillowImageConverter


class TestPillowImageConverter:
    """Test PillowImageConverter infrastructure implementation."""

    @pytest.fixture
    def converter(self):
        """Create PillowImageConverter instance.

        :return: PillowImageConverter instance
        """
        return PillowImageConverter()

    def test_convert_heic_to_jpg_creates_output_folder(self, converter, tmp_path):
        """Test that output folder is created if it doesn't exist.

        :param converter: PillowImageConverter instance
        :param tmp_path: Temporary directory
        :return: None
        """
        input_folder = tmp_path / "input"
        output_folder = tmp_path / "output"
        input_folder.mkdir()

        converter.convert_heic_to_jpg(input_folder, output_folder)

        assert output_folder.exists()

    @patch("src.infrastructure.image_converter_impl.pillow_heif")
    @patch("src.infrastructure.image_converter_impl.Image")
    def test_convert_heic_to_jpg_processes_heic_files(
        self, mock_image_class, mock_pillow_heif, converter, tmp_path
    ):
        """Test that HEIC files are processed correctly.

        :param mock_image_class: Mocked PIL Image class
        :param mock_pillow_heif: Mocked pillow_heif module
        :param converter: PillowImageConverter instance
        :param tmp_path: Temporary directory
        :return: None
        """
        input_folder = tmp_path / "input"
        output_folder = tmp_path / "output"
        input_folder.mkdir()

        # Create a mock HEIC file
        heic_file = input_folder / "test.HEIC"
        heic_file.touch()

        # Mock the HEIF file object
        mock_heif = MagicMock()
        mock_heif.mode = "RGB"
        mock_heif.size = (640, 480)
        mock_heif.data = b"fake_image_data"
        mock_pillow_heif.read_heif.return_value = mock_heif

        # Mock the PIL Image
        mock_image = MagicMock()
        mock_image_class.frombytes.return_value = mock_image

        converter.convert_heic_to_jpg(input_folder, output_folder)

        # Verify HEIC file was read
        mock_pillow_heif.read_heif.assert_called_once()

        # Verify image was created from bytes
        mock_image_class.frombytes.assert_called_once_with(
            "RGB", (640, 480), b"fake_image_data"
        )

        # Verify image was saved as JPEG
        mock_image.save.assert_called_once()
        save_call_args = mock_image.save.call_args
        assert str(save_call_args[0][0]).endswith("test.jpg")
        assert save_call_args[1]["format"] == "JPEG"

    @patch("src.infrastructure.image_converter_impl.pillow_heif")
    @patch("src.infrastructure.image_converter_impl.Image")
    def test_convert_heic_to_jpg_handles_multiple_files(
        self, mock_image_class, mock_pillow_heif, converter, tmp_path
    ):
        """Test converting multiple HEIC files.

        :param mock_image_class: Mocked PIL Image class
        :param mock_pillow_heif: Mocked pillow_heif module
        :param converter: PillowImageConverter instance
        :param tmp_path: Temporary directory
        :return: None
        """
        input_folder = tmp_path / "input"
        output_folder = tmp_path / "output"
        input_folder.mkdir()

        # Create multiple HEIC files
        (input_folder / "image1.HEIC").touch()
        (input_folder / "image2.HEIC").touch()
        (input_folder / "image3.HEIC").touch()

        # Mock the conversion process
        mock_heif = MagicMock()
        mock_heif.mode = "RGB"
        mock_heif.size = (640, 480)
        mock_heif.data = b"data"
        mock_pillow_heif.read_heif.return_value = mock_heif
        mock_image_class.frombytes.return_value = MagicMock()

        converter.convert_heic_to_jpg(input_folder, output_folder)

        # Should have processed all 3 files
        assert mock_pillow_heif.read_heif.call_count == 3

    def test_convert_heic_to_jpg_ignores_non_heic_files(self, converter, tmp_path):
        """Test that non-HEIC files are ignored.

        :param converter: PillowImageConverter instance
        :param tmp_path: Temporary directory
        :return: None
        """
        input_folder = tmp_path / "input"
        output_folder = tmp_path / "output"
        input_folder.mkdir()

        # Create non-HEIC files
        (input_folder / "image.jpg").touch()
        (input_folder / "image.png").touch()
        (input_folder / "document.txt").touch()

        # Should not raise any errors
        converter.convert_heic_to_jpg(input_folder, output_folder)

        # Output folder should be empty (no conversions)
        output_files = list(output_folder.glob("*"))
        assert len(output_files) == 0

    @patch("src.infrastructure.image_converter_impl.pillow_heif")
    def test_convert_heic_to_jpg_handles_conversion_error(
        self, mock_pillow_heif, converter, tmp_path
    ):
        """Test error handling during conversion.

        :param mock_pillow_heif: Mocked pillow_heif module
        :param converter: PillowImageConverter instance
        :param tmp_path: Temporary directory
        :return: None
        """
        input_folder = tmp_path / "input"
        output_folder = tmp_path / "output"
        input_folder.mkdir()

        heic_file = input_folder / "corrupted.HEIC"
        heic_file.touch()

        # Simulate a conversion error
        mock_pillow_heif.read_heif.side_effect = Exception("Corrupted file")

        # Should raise the exception
        with pytest.raises(Exception, match="Corrupted file"):
            converter.convert_heic_to_jpg(input_folder, output_folder)

    def test_convert_heic_to_jpg_with_nested_output_folder(self, converter, tmp_path):
        """Test creating nested output folder structure.

        :param converter: PillowImageConverter instance
        :param tmp_path: Temporary directory
        :return: None
        """
        input_folder = tmp_path / "input"
        output_folder = tmp_path / "output" / "nested" / "path"
        input_folder.mkdir()

        converter.convert_heic_to_jpg(input_folder, output_folder)

        # Nested folders should be created
        assert output_folder.exists()
        assert output_folder.is_dir()

    def test_convert_heic_to_jpg_with_empty_input_folder(self, converter, tmp_path):
        """Test conversion with empty input folder.

        :param converter: PillowImageConverter instance
        :param tmp_path: Temporary directory
        :return: None
        """
        input_folder = tmp_path / "empty_input"
        output_folder = tmp_path / "output"
        input_folder.mkdir()

        # Should handle gracefully
        converter.convert_heic_to_jpg(input_folder, output_folder)

        # Output folder should exist but be empty
        assert output_folder.exists()
        assert len(list(output_folder.glob("*"))) == 0
