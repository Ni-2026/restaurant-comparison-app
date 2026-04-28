"""
scraper_api.py
--------------
VERSION 1: Yelp Fusion API
Fetches live restaurant data via direct API calls.

Pros: Fast, reliable, clean JSON data, no bot detection issues
Cons: Requires a free API key from https://fusion.yelp.com, 5000 calls/day limit

Setup:
  1. Get a free API key at https://fusion.yelp.com
  2. Set it as an environment variable:
       export YELP_API_KEY="your_key_here"
     OR pass it directly: scrape_yelp(api_key="your_key_here")
"""

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────────

YELP_SEARCH_URL  = "https://api.yelp.com/v3/businesses/search"
YELP_DETAILS_URL = "https://api.yelp.com/v3/businesses/{id}"
MAX_PER_PAGE     = 50
REQUEST_TIMEOUT  = 10
RATE_LIMIT_PAUSE = 0.25

PRICE_MAP = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}


# ── Auth ───────────────────────────────────────────────────────────────────────

def _get_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("YELP_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "Yelp API key not found.\n"
            "  Option 1: export YELP_API_KEY='your_key_here'\n"
            "  Option 2: scrape_yelp(..., api_key='your_key_here')\n"
            "  Get a free key at https://fusion.yelp.com"
        )
    return key

def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch_page(
    location: str,
    cuisine: str,
    budget: str,
    offset: int,
    limit: int,
    api_key: str,
) -> list[dict]:
    """Calls the Yelp Business Search endpoint for one page of results."""
    params = {
        "term":     cuisine,
        "location": location,
        "limit":    min(limit, MAX_PER_PAGE),
        "offset":   offset,
        "sort_by":  "best_match",
    }
    if budget and budget in PRICE_MAP:
        params["price"] = str(PRICE_MAP[budget])

    try:
        response = requests.get(
            YELP_SEARCH_URL,
            headers=_headers(api_key),
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 401:
            raise PermissionError("Invalid or expired Yelp API key.")
        if response.status_code == 429:
            logger.warning("Rate limit hit — waiting 2 seconds.")
            time.sleep(2)
            return fetch_page(location, cuisine, budget, offset, limit, api_key)
        if response.status_code == 400:
            logger.error("Bad request: %s", response.json().get("description", "unknown"))
            return []
        response.raise_for_status()
        return response.json().get("businesses", [])

    except requests.exceptions.ConnectionError:
        logger.error("Network error: could not reach Yelp API.")
        return []
    except requests.exceptions.Timeout:
        logger.error("Request timed out after %ds.", REQUEST_TIMEOUT)
        return []
    except requests.exceptions.RequestException as exc:
        logger.error("API request failed: %s", exc)
        return []


def fetch_business_details(business_id: str, api_key: str) -> dict:
    """Fetches full details (hours, photos) for a single business."""
    url = YELP_DETAILS_URL.format(id=business_id)
    try:
        response = requests.get(url, headers=_headers(api_key), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.warning("Could not fetch details for %s: %s", business_id, exc)
        return {}


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_business(biz: dict) -> Optional[dict]:
    """Normalizes a raw Yelp API business object into a flat restaurant dict."""
    if not biz or not biz.get("name"):
        return None
    if biz.get("is_closed", False):
        return None

    categories = biz.get("categories", [])
    category_str = ", ".join(c.get("title", "") for c in categories) or "Unknown"

    loc = biz.get("location", {})
    address = ", ".join(
        p for p in [
            loc.get("address1", ""),
            loc.get("city", ""),
            loc.get("state", ""),
            loc.get("zip_code", ""),
        ] if p
    )

    coords = biz.get("coordinates", {})

    return {
        "name":      biz.get("name", "").strip(),
        "rating":    float(biz.get("rating", 0.0)),
        "reviews":   int(biz.get("review_count", 0)),
        "price":     biz.get("price", "N/A"),
        "category":  category_str,
        "url":       biz.get("url", ""),
        "address":   address,
        "phone":     biz.get("display_phone", ""),
        "latitude":  coords.get("latitude"),
        "longitude": coords.get("longitude"),
        "yelp_id":   biz.get("id", ""),
    }


# ── Entry Point ────────────────────────────────────────────────────────────────

def scrape_yelp(
    location: str,
    cuisine: str,
    budget: str = "",
    max_results: int = 10,
    api_key: Optional[str] = None,
    fetch_details: bool = False,
) -> list[dict]:
    """
    Fetches live restaurant data from the Yelp Fusion API.

    Args:
        location:       City/neighborhood (e.g. "Honolulu, HI").
        cuisine:        Food keyword (e.g. "ramen", "tacos").
        budget:         Price filter: "$", "$$", "$$$", "$$$$", or "" for any.
        max_results:    Max restaurants to return.
        api_key:        Yelp API key (falls back to YELP_API_KEY env var).
        fetch_details:  If True, makes an extra call per restaurant for hours.

    Returns:
        List of flat restaurant dicts.
    """
    if not location or not cuisine:
        raise ValueError("Both 'location' and 'cuisine' must be non-empty.")

    key = _get_api_key(api_key)
    all_restaurants = []
    offset = 0

    logger.info("Fetching up to %d '%s' restaurants in '%s'...", max_results, cuisine, location)

    while len(all_restaurants) < max_results:
        remaining = max_results - len(all_restaurants)
        raw = fetch_page(location, cuisine, budget, offset, min(remaining, MAX_PER_PAGE), key)

        if not raw:
            break

        for biz in raw:
            parsed = parse_business(biz)
            if not parsed:
                continue

            if fetch_details and parsed["yelp_id"]:
                details = fetch_business_details(parsed["yelp_id"], key)
                hours_info = details.get("hours", [])
                if hours_info:
                    parsed["is_open_now"] = hours_info[0].get("is_open_now")
                time.sleep(RATE_LIMIT_PAUSE)

            all_restaurants.append(parsed)
            if len(all_restaurants) >= max_results:
                break

        offset += len(raw)
        time.sleep(RATE_LIMIT_PAUSE)

        if offset >= 1000:
            break

    logger.info("Done. Fetched %d restaurants.", len(all_restaurants))
    return all_restaurants
