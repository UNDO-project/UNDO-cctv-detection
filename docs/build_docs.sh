#!/usr/bin/env bash
# Build Sphinx documentation for CCTV Detection System

set -e

# Change to docs directory
cd "$(dirname "$0")"

echo "Building Sphinx documentation..."
echo "================================"

# Clean previous build
echo "Cleaning previous build..."
rm -rf build/html

# Copy markdown guides to source directory
echo "Copying markdown guides to source..."
cp -f {ARCHITECTURE,CONFIGURATION,TRAINING,EVALUATION,UI_GUIDE,DATASET,DEVELOPMENT}.md source/

# Build HTML documentation
echo "Building HTML documentation..."
uv run sphinx-build -b html source build/html

echo ""
echo "Build complete!"
echo "Documentation is available at: build/html/index.html"
echo ""
echo "To view locally:"
echo "  open build/html/index.html  # macOS"
echo "  xdg-open build/html/index.html  # Linux"