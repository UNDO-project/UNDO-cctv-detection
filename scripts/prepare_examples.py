"""Copy representative images to examples/ directory for Gradio UI.

This script copies a selection of validation images to the examples directory
to be used as demo images in the Gradio interface.
"""

import shutil

from src.config import settings


def main():
    """Copy sample images to examples directory.

    :return: None
    """
    examples_dir = settings.paths.project_root / "examples"
    examples_dir.mkdir(exist_ok=True)

    # Source: validation images
    val_images = settings.paths.ultralytics_dir / "images" / "val"

    if not val_images.exists():
        print(f"❌ No validation images found at {val_images}")
        print("Please ensure the dataset is properly prepared.")
        return

    # Get all images and sort for consistency
    all_images = sorted(list(val_images.glob("*.jpg")) + list(val_images.glob("*.png")))

    if not all_images:
        print(f"❌ No images found in {val_images}")
        return

    # Copy first 6 images as examples
    num_examples = min(6, len(all_images))

    for i, img_path in enumerate(all_images[:num_examples], 1):
        dest = examples_dir / f"example_{i}{img_path.suffix}"
        shutil.copy(img_path, dest)
        print(f"✅ Copied: {img_path.name} -> {dest.name}")

    print(f"\n✅ {num_examples} example images prepared in {examples_dir}")
    print("   Run 'uv run python app.py' to launch the UI with examples.")


if __name__ == "__main__":
    main()
