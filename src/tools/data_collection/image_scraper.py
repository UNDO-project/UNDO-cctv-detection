from pathlib import Path
from typing import List

from loguru import logger
from playwright.sync_api import sync_playwright

from src.config import settings
from src.domain.camera import CameraDataFromCsv


class ImageScraper:
    """Web scraper for capturing camera images using Playwright.

    Uses configuration from settings for timeouts and cookie dialog handling.
    """

    def __init__(
        self, output_dir: Path | None = None, headless: bool | None = None
    ) -> None:
        """Initialize image scraper.

        :param output_dir: Output directory for screenshots (defaults to settings.scraper.output_dir)
        :param headless: Run browser in headless mode (defaults to settings.scraper.headless)
        """
        self.output_dir = output_dir or settings.scraper.output_dir
        self.headless = headless if headless is not None else settings.scraper.headless

    def scrape_images(self, cameras: List[CameraDataFromCsv]) -> None:
        """
        Capture screenshots from given URLs and save them with structured filenames.
        :param cameras: List of urls from csv file
        :return:
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cookies_rejected = False  # Track if cookies have already been rejected

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context()
            page = context.new_page()

            for camera in cameras:
                filename = (
                    self.output_dir / f"camera_{camera.latitude}_{camera.longitude}.png"
                )
                try:
                    page.goto(camera.url, timeout=settings.scraper.browser_timeout_ms)

                    if not cookies_rejected:
                        # XPath to match buttons with "Reject all" labels
                        reject_button_xpath = (
                            f"//button[@aria-label='{settings.scraper.reject_all_text}' "
                            f"or @aria-label='{settings.scraper.reject_all_text_gr}']"
                        )
                        try:
                            # Wait for the "Reject all" button (if it exists)
                            page.wait_for_selector(
                                reject_button_xpath,
                                timeout=settings.scraper.cookie_dialog_timeout_ms,
                            )
                            # Click the "Reject all" button
                            page.click(reject_button_xpath)
                            # Allow the page to load after rejecting cookies
                            page.wait_for_load_state("networkidle")
                            cookies_rejected = True  # Mark cookies as rejected
                            logger.info("Cookies rejected successfully.")
                        except TimeoutError:
                            logger.debug(
                                "Cookie consent dialog not found (likely not shown)"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Unexpected error handling cookie consent: {e}"
                            )
                            raise

                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(settings.scraper.page_settle_timeout_ms)

                    page.screenshot(path=str(filename))
                    logger.success(f"Saved: {filename}")
                except Exception as e:
                    logger.error(f"Failed: {camera.url} - {e}")

            browser.close()
