CCTV Detection System Documentation
====================================

Welcome to the CCTV Detection System documentation. This system provides multi-model object detection for identifying CCTV cameras and signage using YOLOv8, Faster R-CNN, and DETR models.

.. image:: https://img.shields.io/badge/license-CC0%201.0-blue
   :alt: License

.. image:: https://img.shields.io/badge/python-3.11+-green
   :alt: Python Version

.. image:: https://img.shields.io/badge/tests-passing-brightgreen
   :alt: Tests

This project is part of the `Understanding Nordic Digital Order (UNDO) <https://undo-project.info>`_ project.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Install uv package manager
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --all-extras

   # Install pre-commit hooks
   uv run pre-commit install

Launch Application
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start the Gradio web interface
   uv run python app.py

   # Or use the console script
   uv run cctv-ui

Visit http://127.0.0.1:7860 in your browser.

User Guides
-----------

Comprehensive guides for using, training, and developing with the CCTV Detection System.

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   ARCHITECTURE
   CONFIGURATION
   TRAINING
   EVALUATION
   UI_GUIDE
   DATASET
   DEVELOPMENT

API Reference
-------------

Detailed API documentation for all modules and classes.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   modules

Architecture Overview
---------------------

The project follows a clean layered architecture:

* **Domain Layer** (``src/domain/``): Core business logic and interfaces
* **Infrastructure Layer** (``src/infrastructure/``): Concrete implementations
* **Application Layer** (``src/application/``): Use case orchestration
* **UI Layer** (``src/ui/``): Gradio web interface
* **Scripts Layer** (``scripts/``): Command-line utilities

See the :doc:`ARCHITECTURE` guide for detailed information.

Key Features
------------

* 🎯 **Multi-Model Support**: Compare YOLOv8, Faster R-CNN, and DETR
* 🎨 **Interactive Web UI**: Gradio-based interface with model comparison
* 📊 **Performance Dashboard**: Training metrics and analytics
* 🧪 **Comprehensive Testing**: >65% code coverage
* ⚙️ **Type-Safe Configuration**: Pydantic v2 settings

Console Scripts
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``uv run cctv-ui``
     - Launch Gradio web interface
   * - ``uv run cctv-train``
     - Train YOLOv8 model
   * - ``uv run cctv-train-faster-rcnn``
     - Train Faster R-CNN model
   * - ``uv run cctv-train-detr``
     - Train DETR model
   * - ``uv run cctv-evaluate-faster-rcnn``
     - Evaluate Faster R-CNN and compute mAP
   * - ``uv run cctv-evaluate-detr``
     - Evaluate DETR and compute mAP
   * - ``uv run cctv-prepare-examples``
     - Prepare example images for UI
   * - ``uv run cctv-benchmark``
     - Benchmark inference speed

Contributing
------------

Contributions are welcome! Please see the :doc:`DEVELOPMENT` guide for:

* Setting up your development environment
* Running tests and code quality checks
* Coding standards and best practices
* Submitting pull requests

License
-------

CC0 1.0 Universal (Public Domain)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`