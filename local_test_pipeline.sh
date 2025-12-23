#!/bin/bash
PYTHONPATH="${PYTHONPATH}:$(realpath "./src")"
export PYTHONPATH

# Navigate to the script's directory (project root)
cd "$(dirname "$0")" || exit

echo "Running application layer tests"
echo "----Running camera image downloader tests"
pytest tests/application/test_camera_image_downloader.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running dataset preparation tests"
pytest tests/application/test_dataset_preparation.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running surveillance service tests"
pytest tests/application/test_surveillance_service.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running training service tests"
pytest tests/application/test_training_service.py --no-cov
echo "Done..."
echo "==============================================="

echo "Running domain layer tests"
echo "----Running camera tests"
pytest tests/domain/test_camera.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running distance calculator tests"
pytest tests/domain/services/test_distance_calculator.py --no-cov
echo "Done..."
echo "==============================================="

echo "Running infrastructure layer tests"
echo "----Running data loaders tests"
pytest tests/infrastructure/test_data_loaders.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running dataset preparer tests"
pytest tests/infrastructure/test_dataset_preparer.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running image converter tests"
pytest tests/infrastructure/test_image_converter_impl.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running splitter tests"
pytest tests/infrastructure/test_splitters.py --no-cov
echo "Done..."
echo "==============================================="

echo "Running presentation layer tests"
echo "----Running main ui tests"
pytest tests/presentation/test_main_ui.py --no-cov
echo "Done..."
echo "==============================================="

echo "Running split data tests"
pytest tests/test_split_data.py --no-cov
echo "Done..."
echo "==============================================="

echo ""
echo "=========================================="
echo "Running full test suite with coverage"
echo "=========================================="
pytest tests/