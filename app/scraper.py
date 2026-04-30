# scraper.py
# Nizhen He — ITM352 Restaurant Comparison App
# Scrapes Yelp search results using Selenium (page load) + BeautifulSoup (parsing)

import time
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException


# ── Driver Setup ──────────────────────────────────────────────────────────────

def create_driver():
    """
    Creates a headless Chrome WebDriver with anti-detection options.
    Requires: Google Chrome installed + chromedriver matching your Chrome version.
    Install chromedriver: pip install webdriver-manager
    """
    options = Options()
    options.add_argument("--headless")               # Run without opening browser window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    # webdriver-manager auto-downloads the right chromedriver for your Chrome
    from webdriver_manager.chrome import ChromeDriverManager
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    # Mask the navigator.webdriver flag so Yelp doesn't detect automation
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
    )
    return driver


# ── URL Builder ───────────────────────────────────────────────────────────────

def build_yelp_url(location: str, cuisine: str, budget: int) -> str:
    """
    Builds a Yelp search URL from user inputs.

    Args:
        location (str): e.g. "Honolulu, HI"
        cuisine  (str): e.g. "japanese"
        budget   (int): 1=$  2=$$  3=$$$  4=$$$$

    Returns:
        str: Full Yelp search URL
    """
    base = "https://www.yelp.com/search"
    # Encode spaces as + for URL
    loc_enc     = location.strip().replace(" ", "+")
    cuisine_enc = cuisine.strip().replace(" ", "+")
    url = f"{base}?find_desc={cuisine_enc}+restaurants&find_loc={loc_enc}&attrs=RestaurantsPriceRange2:{budget}"
    return url


# ── Main Scrape Function ──────────────────────────────────────────────────────

def scrape_yelp(location: str, cuisine: str, budget: int, max_results: int = 10) -> list[dict]:
    """
    Scrapes Yelp for restaurant listings matching the given inputs.

    Args:
        location   (str): City/address to search
        cuisine    (str): Type of cuisine
        budget     (int): Price range 1–4
        max_results(int): Max number of restaurants to return (default 10)

    Returns:
        list[dict]: Each dict has keys:
            name, rating, review_count, price, address, phone, yelp_url
        Returns empty list if scraping fails.
    """
    url = build_yelp_url(location, cuisine, budget)
    print(f"[Scraper] Fetching: {url}")

    driver = create_driver()
    restaurants = []

    try:
        driver.get(url)

        # Wait until at least one result card is visible (up to 15 sec)
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="serp-ia-card"]'))
            )
        except TimeoutException:
            print("[Scraper] Warning: Timed out waiting for results. Page may have changed.")

        # Random delay to mimic human behavior
        time.sleep(random.uniform(2.0, 4.0))

        # Hand the fully-loaded page HTML to BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        restaurants = parse_results(soup, max_results)

    except WebDriverException as e:
        print(f"[Scraper] WebDriver error: {e}")

    finally:
        driver.quit()  # Always close the browser

    print(f"[Scraper] Found {len(restaurants)} restaurants.")
    return restaurants


# ── HTML Parser ───────────────────────────────────────────────────────────────

def parse_results(soup: BeautifulSoup, max_results: int) -> list[dict]:
    """
    Parses BeautifulSoup HTML to extract restaurant data from Yelp result cards.

    NOTE: Yelp changes its HTML structure periodically.
    If this breaks, inspect the page with DevTools and update the selectors below.

    Args:
        soup       (BeautifulSoup): Parsed page HTML
        max_results(int): Cap on how many results to return

    Returns:
        list[dict]: Cleaned raw restaurant records
    """
    restaurants = []

    # Each result card has this test ID — most stable selector on Yelp
    cards = soup.find_all("div", attrs={"data-testid": "serp-ia-card"})

    # Fallback: try the older class-based selector if test IDs aren't present
    if not cards:
        cards = soup.find_all("li", class_=lambda c: c and "businessList" in c)

    for card in cards[:max_results]:
        try:
            restaurant = extract_card_data(card)
            if restaurant:
                restaurants.append(restaurant)
        except Exception as e:
            # Skip malformed cards rather than crashing the whole scrape
            print(f"[Parser] Skipped a card due to error: {e}")
            continue

    return restaurants


def extract_card_data(card) -> dict | None:
    """
    Extracts fields from a single Yelp result card element.

    Args:
        card: BeautifulSoup Tag for one restaurant card

    Returns:
        dict with restaurant fields, or None if critical fields are missing
    """
    # ── Name ──────────────────────────────────────────────────────────────────
    name_tag = card.find("a", attrs={"data-testid": "biz-name"})
    if not name_tag:
        return None  # Can't use a result with no name
    name = name_tag.get_text(strip=True)

    # ── Yelp URL ───────────────────────────────────────────────────────────────
    href = name_tag.get("href", "")
    yelp_url = "https://www.yelp.com" + href if href.startswith("/") else href

    # ── Rating (e.g. "4.5 star rating") ───────────────────────────────────────
    rating = None
    rating_tag = card.find("span", attrs={"data-testid": "rating-stars"})
    if rating_tag:
        aria = rating_tag.get("aria-label", "")          # "4.5 star rating"
        rating = parse_float(aria.split(" ")[0])

    # ── Review Count (e.g. "312 reviews") ─────────────────────────────────────
    review_count = 0
    review_tag = card.find("span", attrs={"data-testid": "review-count"})
    if review_tag:
        review_count = parse_int(review_tag.get_text(strip=True))

    # ── Price (e.g. "$$") ─────────────────────────────────────────────────────
    price = None
    price_tag = card.find("span", attrs={"class": lambda c: c and "priceRange" in " ".join(c)})
    if not price_tag:
        # Fallback: look for $/$$/$$$ text pattern directly
        price_tag = card.find(string=lambda t: t and t.strip() in ["$", "$$", "$$$", "$$$$"])
    price = price_tag.get_text(strip=True) if price_tag else "N/A"

    # ── Address ────────────────────────────────────────────────────────────────
    address = "N/A"
    address_tag = card.find("address")
    if address_tag:
        address = address_tag.get_text(separator=" ", strip=True)

    # ── Phone (not always present on search page) ──────────────────────────────
    phone = "N/A"
    phone_tag = card.find("p", attrs={"data-testid": "biz-phone"})
    if phone_tag:
        phone = phone_tag.get_text(strip=True)

    return {
        "name":         name,
        "rating":       rating,
        "review_count": review_count,
        "price":        price,
        "address":      address,
        "phone":        phone,
        "yelp_url":     yelp_url,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_float(value: str) -> float | None:
    """Safely converts a string to float. Returns None if it fails."""
    try:
        return float(value.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def parse_int(value: str) -> int:
    """Safely converts a string to int, stripping non-numeric chars."""
    try:
        cleaned = "".join(filter(str.isdigit, value))
        return int(cleaned) if cleaned else 0
    except (ValueError, AttributeError):
        return 0