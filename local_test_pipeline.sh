#!/bin/bash
PYTHONPATH="${PYTHONPATH}:$(realpath "./src")"
export PYTHONPATH

# Navigate to the script's directory (project root)
cd "$(dirname "$0")" || exit

echo "Running application layer tests"

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

echo "----Running DETR dataset preparer tests"
pytest tests/infrastructure/test_detr_dataset.py --no-cov
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

echo "----Running device selector tests"
pytest tests/infrastructure/test_device_selector.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running trainer tests"
pytest tests/infrastructure/test_trainers.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running DETR trainer tests"
pytest tests/infrastructure/test_detr_trainer.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running detectors tests"
pytest tests/infrastructure/test_detectors.py --no-cov
echo "Done..."
echo "==============================================="

echo "Running UI layer tests"
echo "----Running gradio app tests"
pytest tests/ui/test_gradio_app.py --no-cov
echo "Done..."
echo "==============================================="

echo "----Running camera image downloader tests"
pytest tests/tools/data_collection/test_camera_image_downloader.py --no-cov
echo "Done..."
echo "==============================================="

echo "Running configuration settings tests"
pytest tests/test_config.py --no-cov
echo "Done..."
echo "==============================================="

echo ""
echo "=========================================="
echo "Running full test suite with coverage"
echo "=========================================="
pytest tests/