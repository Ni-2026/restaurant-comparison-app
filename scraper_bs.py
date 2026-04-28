"""
scraper_bs.py
-------------
VERSION 2: Selenium + BeautifulSoup
Scrapes live Yelp search results using a real browser + HTML parsing.

Pros: No API key, no call limits, works for anyone without signup
Cons: Slower, can break if Yelp updates their HTML layout, against Yelp ToS,
      requires ChromeDriver installed and matching your Chrome version.

Setup:
  1. Install dependencies:
       pip install selenium beautifulsoup4
  2. Install ChromeDriver:
       https://chromedriver.chromium.org/downloads
       Make sure the version matches your installed Chrome browser.
  3. No API key or account needed — just run it.
"""

import logging
import random
import time
from typing import Optional

from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────────

YELP_BASE_URL        = "https://www.yelp.com/search"
RESULTS_PER_PAGE     = 10
PAGE_LOAD_TIMEOUT    = 15    # seconds to wait for cards to appear
REQUEST_DELAY        = (2, 4) # random sleep between pages to avoid blocks

# Yelp CSS selectors — update these if Yelp changes their layout
SELECTORS = {
    "result_card":    '[data-testid="serp-ia-card"]',
    "name":           'a[data-testid="biz-name"]',
    "rating":         'div[aria-label*="star rating"]',
    "review_count":   'span[class*="reviewCount"]',
    "price":          'span[class*="priceRange"]',
    "category":       'span[class*="css-chan6m"]',
    "address":        'address',
    "hours":          'span[class*="secondaryAttributes"]',
}


# ── Driver Setup ───────────────────────────────────────────────────────────────

def _build_driver() -> "webdriver.Chrome":
    """
    Launches a headless Chrome browser that looks as human as possible.
    Raises ImportError if selenium is not installed.
    """
    if not SELENIUM_AVAILABLE:
        raise ImportError(
            "selenium is not installed.\n"
            "  Run: pip install selenium\n"
            "  Also install ChromeDriver: https://chromedriver.chromium.org/downloads"
        )

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)

    # Hide the webdriver flag that sites use to detect bots
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


# ── HTML Parsing ───────────────────────────────────────────────────────────────

def parse_restaurant_html(html: str) -> list[dict]:
    """
    Parses a fully-rendered Yelp search results page.

    Args:
        html: Raw HTML string from Selenium's driver.page_source.

    Returns:
        List of restaurant dicts. Empty list if no cards found or parse fails.
    """
    restaurants = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select(SELECTORS["result_card"])

        if not cards:
            logger.warning(
                "No result cards found — Yelp may have updated their layout. "
                "Check SELECTORS in scraper_bs.py."
            )
            return []

        for card in cards:
            parsed = _parse_card(card)
            if parsed:
                restaurants.append(parsed)

    except Exception as exc:
        logger.error("Error parsing HTML: %s", exc)

    return restaurants


def _parse_card(card: "BeautifulSoup") -> Optional[dict]:
    """Extracts all fields from a single Yelp result card."""
    try:
        # ── Name + URL ──
        name_tag = card.select_one(SELECTORS["name"])
        if not name_tag:
            return None
        name = name_tag.get_text(strip=True)
        relative_url = name_tag.get("href", "")
        url = "https://www.yelp.com" + relative_url if relative_url.startswith("/") else relative_url

        # ── Rating (from aria-label like "4.5 star rating") ──
        rating = 0.0
        rating_tag = card.select_one(SELECTORS["rating"])
        if rating_tag:
            aria = rating_tag.get("aria-label", "")
            try:
                rating = float(aria.split()[0])
            except (ValueError, IndexError):
                pass

        # ── Review count ──
        reviews = 0
        review_tag = card.select_one(SELECTORS["review_count"])
        if review_tag:
            try:
                reviews = int(review_tag.get_text(strip=True).replace(",", ""))
            except ValueError:
                pass

        # ── Price ──
        price_tag = card.select_one(SELECTORS["price"])
        price = price_tag.get_text(strip=True) if price_tag else "N/A"

        # ── Category ──
        cat_tags = card.select(SELECTORS["category"])
        category = cat_tags[0].get_text(strip=True) if cat_tags else "Unknown"

        # ── Address ──
        address_tag = card.select_one(SELECTORS["address"])
        address = address_tag.get_text(strip=True) if address_tag else ""

        return {
            "name":     name,
            "rating":   rating,
            "reviews":  reviews,
            "price":    price,
            "category": category,
            "address":  address,
            "url":      url,
        }

    except Exception as exc:
        logger.warning("Skipping card — parse error: %s", exc)
        return None


# ── Pagination ─────────────────────────────────────────────────────────────────

def handle_pagination(
    driver: "webdriver.Chrome",
    base_url: str,
    max_results: int,
) -> list[str]:
    """
    Navigates through Yelp result pages and collects rendered HTML.

    Stops when max_results have been seen or no more pages exist.

    Args:
        driver:      Active Selenium WebDriver.
        base_url:    First-page Yelp search URL.
        max_results: Stop collecting after this many results.

    Returns:
        List of raw HTML strings (one per page visited).
    """
    pages_html = []
    collected  = 0
    start      = 0

    while collected < max_results:
        url = f"{base_url}&start={start}" if start > 0 else base_url
        logger.info("Loading page (offset %d): %s", start, url)

        try:
            driver.get(url)

            # Wait for result cards to appear in the DOM
            WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["result_card"])
                )
            )

            # Extra wait for any lazy-loaded content
            time.sleep(random.uniform(*REQUEST_DELAY))

            html = driver.page_source
            pages_html.append(html)

            # Count how many cards are on this page
            soup = BeautifulSoup(html, "html.parser")
            cards_on_page = len(soup.select(SELECTORS["result_card"]))

            logger.info("Found %d cards on this page.", cards_on_page)

            if cards_on_page == 0:
                logger.info("No more results — stopping pagination.")
                break

            collected += cards_on_page
            start     += RESULTS_PER_PAGE

        except TimeoutException:
            logger.warning("Timed out waiting for results at offset %d.", start)
            break
        except WebDriverException as exc:
            logger.error("WebDriver error: %s", exc)
            break

    return pages_html


# ── Entry Point ────────────────────────────────────────────────────────────────

def scrape_yelp(
    location: str,
    cuisine: str,
    budget: str = "",
    max_results: int = 10,
) -> list[dict]:
    """
    Scrapes live Yelp search results using a real headless Chrome browser.

    No API key required. No account needed. Just ChromeDriver installed.

    Args:
        location:    City/neighborhood (e.g. "Honolulu, HI").
        cuisine:     Food keyword (e.g. "ramen", "tacos").
        budget:      Price filter: "$", "$$", "$$$", "$$$$", or "" for any.
        max_results: Max number of restaurants to return.

    Returns:
        List of flat restaurant dicts:
          {name, rating, reviews, price, category, address, url}
        Returns an empty list if scraping fails or Yelp blocks the request.

    Raises:
        ValueError:  If location or cuisine is empty.
        ImportError: If selenium is not installed.

    Notes:
        - Yelp actively detects and blocks scrapers. This may stop working
          at any time if Yelp updates their anti-bot measures or HTML layout.
        - Scraping Yelp is against their Terms of Service (Section 10).
          Use for educational/personal projects only.
        - If results come back empty, Yelp may have updated their CSS selectors.
          Check and update the SELECTORS dict at the top of this file.
    """
    if not location or not cuisine:
        raise ValueError("Both 'location' and 'cuisine' must be non-empty.")

    # Build the Yelp search URL
    query    = cuisine.replace(" ", "+")
    loc      = location.replace(" ", "+")
    base_url = f"{YELP_BASE_URL}?find_desc={query}&find_loc={loc}"

    # Add price filter if specified
    if budget:
        price_codes = {"$": "1", "$$": "2", "$$$": "3", "$$$$": "4"}
        code = price_codes.get(budget)
        if code:
            base_url += f"&attrs=RestaurantsPriceRange2.{code}"

    logger.info("Starting scrape for '%s' in '%s' (budget: %s)...",
                cuisine, location, budget or "any")

    driver = None
    all_restaurants = []

    try:
        driver = _build_driver()
        pages_html = handle_pagination(driver, base_url, max_results)

        for html in pages_html:
            parsed = parse_restaurant_html(html)
            all_restaurants.extend(parsed)
            if len(all_restaurants) >= max_results:
                break

    except ImportError:
        raise
    except Exception as exc:
        logger.error("Scraping failed: %s", exc)

    finally:
        if driver:
            driver.quit()
            logger.info("Browser closed.")

    # Deduplicate by name and trim to requested count
    seen, unique = set(), []
    for r in all_restaurants:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)
        if len(unique) >= max_results:
            break

    logger.info("Done. Scraped %d unique restaurants.", len(unique))
    return unique
