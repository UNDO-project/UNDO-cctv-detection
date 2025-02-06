from pathlib import Path
from typing import Tuple, List

from loguru import logger
from playwright.sync_api import sync_playwright

from src.config import REJECT_ALL, REJECT_ALL_GR


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
    cookies_rejected = False  # Track if cookies have already been rejected

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        for lat, lon, url in cameras:
            filename = output_dir / f"camera_{lat}_{lon}.png"
            try:
                page.goto(url, timeout=20000)  # Allow more time for page navigation

                if not cookies_rejected:
                    # XPath to match buttons with "Reject all"labels
                    reject_button_xpath = (
                        "//button[@aria-label="
                        + REJECT_ALL
                        + " or @aria-label="
                        + REJECT_ALL_GR
                        + "]"
                    )
                    try:
                        # Wait for the "Reject all" button (if it exists)
                        page.wait_for_selector(reject_button_xpath, timeout=5000)
                        # Click the "Reject all" button
                        page.click(reject_button_xpath)
                        # Allow the page to load after rejecting cookies
                        page.wait_for_load_state("networkidle")
                        cookies_rejected = True  # Mark cookies as rejected
                        logger.info("Cookies rejected successfully.")
                    except Exception as e:
                        # If the "Reject all" button is not found, skip
                        logger.info(
                            f"Cookie consent not shown or already rejected. {e}"
                        )

                page.wait_for_load_state(
                    "networkidle"
                )  # Wait for the network to go idle
                page.wait_for_timeout(
                    5000
                )  # Wait for an additional 5 seconds as a buffer

                page.screenshot(path=str(filename))
                logger.success(f"Saved: {filename}")
            except Exception as e:
                logger.error(f"Failed: {url} - {e}")

        browser.close()
