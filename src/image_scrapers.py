from pathlib import Path
from typing import Tuple, List

from loguru import logger
from playwright.sync_api import sync_playwright


def capture_screenshots(
    cameras: List[Tuple[float, float, str]], output_dir: Path
) -> None:
    """
    Capture screenshots from given URLs and save them with structured filenames.
    :param cameras: List of urls from csv file
    :param output_dir: The Path object to the output directory
    :return:
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()

        for lat, lon, url in cameras:
            filename = output_dir / f"camera_{lat}_{lon}.png"
            try:
                page.goto(url, timeout=10000)  # Visit URL
                page.screenshot(path=str(filename))  # Take screenshot
                logger.success(f"Saved: {filename}")
            except Exception as e:
                logger.error(f"Failed: {url} - {e}")

        browser.close()
