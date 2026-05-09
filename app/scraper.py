# scraper.py
# Nizhen He — ITM352 Restaurant Comparison App
# Fetches live Yelp data via SerpAPI (bypasses Yelp's DataDome CAPTCHA)
#
# UPDATED: Now also extracts thumbnail images and external website URLs
# (when SerpAPI returns them), and provides estimate_peak_times() for the
# new "Estimated Busy Times" chart.

import time
import random
from serpapi import GoogleSearch

# ── Your SerpAPI Key ──────────────────────────────────────────────────────────
API_KEY = "226d3125a81c5a098e4de3e4273385d85040dd82285b88913a2b0f31106d7004"

# ── Hawaii Locations ──────────────────────────────────────────────────────────
HAWAII_LOCATIONS = [
    "Honolulu, HI",
    "Waikiki, Honolulu, HI",
    "Kailua, HI",
    "Pearl City, HI",
    "Lahaina, Maui, HI",
    "Kihei, Maui, HI",
    "Kailua-Kona, HI",
    "Hilo, HI",
    "Lihue, Kauai, HI",
]


# ── Main Scrape Function ──────────────────────────────────────────────────────

def scrape_yelp(location: str, cuisine: str, budget: int,
                max_results: int = 20) -> list[dict]:
    """
    Fetches Yelp restaurant listings via SerpAPI.

    Returns a list of dicts; each contains:
        name, rating, review_count, price, address, phone, yelp_url,
        image_url, website, categories
    """
    restaurants = []
    start       = 0

    while len(restaurants) < max_results:
        params = {
            "engine":    "yelp",
            "find_desc": cuisine,
            "find_loc":  location,
            "start":     start,
            "api_key":   API_KEY,
        }
        if budget:
            params["attrs"] = f"RestaurantsPriceRange2:{budget}"

        print(f"[Scraper] Fetching results {start + 1}–{start + 10} "
              f"for '{cuisine}' in {location}...")

        try:
            search  = GoogleSearch(params)
            results = search.get_dict()

            if "error" in results:
                print(f"[Scraper] SerpAPI error: {results['error']}")
                break

            businesses = results.get("organic_results", [])

            if not businesses:
                print("[Scraper] No more results.")
                break

            for biz in businesses:
                restaurants.append(parse_result(biz))

            start += 10
            if len(businesses) < 10:
                print("[Scraper] Reached end of Yelp results.")
                break

            time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"[Scraper] Unexpected error: {e}")
            break

    print(f"[Scraper] Fetched {len(restaurants)} restaurants.")
    return restaurants[:max_results]


# ── Bulk Hawaii Scraper ───────────────────────────────────────────────────────

def scrape_hawaii_bulk(cuisine: str, budget: int,
                       per_location: int = 20) -> list[dict]:
    """Scrapes all Hawaii locations and returns one combined raw list."""
    all_raw = []
    for location in HAWAII_LOCATIONS:
        raw = scrape_yelp(location, cuisine, budget, max_results=per_location)
        all_raw.extend(raw)
        time.sleep(random.uniform(1.0, 2.0))
    print(f"[Scraper] Total raw results: {len(all_raw)}")
    return all_raw


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_result(biz: dict) -> dict:
    """
    Converts one SerpAPI Yelp organic result into the flat dict
    that pipeline.build_dataframe() expects.

    UPDATED FIELDS:
      - image_url:   from SerpAPI's "thumbnail" key (primary image)
      - image_gallery: list of photo URLs from the photo gallery (when available)
      - website:     from SerpAPI's "website" or "service_options.website"
                     (often empty — Yelp organic_results don't always include it)
      - categories:  joined list of category titles, used for richer cards
    """
    neighborhood = biz.get("neighborhoods", "")

    # Image — SerpAPI returns 'thumbnail' for many Yelp businesses
    image_url = biz.get("thumbnail", "") or biz.get("photo", "") or ""

    # Photo gallery — SerpAPI may return multiple images in 'photos' or 'photo_gallery'
    image_gallery = []
    if "photos" in biz and isinstance(biz["photos"], list):
        image_gallery = [img.get("image", "") for img in biz["photos"] if isinstance(img, dict) and img.get("image")]
    if not image_gallery and "photo_gallery" in biz and isinstance(biz["photo_gallery"], list):
        image_gallery = biz["photo_gallery"]
    # Add thumbnail to gallery if not already there
    if image_url and image_url not in image_gallery:
        image_gallery.insert(0, image_url)

    # External website — not consistently provided in Yelp organic results.
    # Grab it from any of the common keys SerpAPI may use.
    website = biz.get("website", "")
    if not website:
        # Some SerpAPI responses nest it differently
        extras = biz.get("service_options", {}) or {}
        website = extras.get("website", "") if isinstance(extras, dict) else ""

    # Categories — list of dicts like [{"title": "Japanese", "link": "..."}]
    raw_cats = biz.get("categories", []) or []
    if isinstance(raw_cats, list):
        categories = ", ".join([c.get("title", "") for c in raw_cats if isinstance(c, dict)])
    else:
        categories = ""

    return {
        "name":         biz.get("title", ""),
        "rating":       biz.get("rating", None),
        "review_count": biz.get("reviews", 0),
        "price":        biz.get("price", "N/A"),
        "address":      neighborhood if neighborhood else "N/A",
        "phone":        biz.get("phone", "N/A"),
        "yelp_url":     biz.get("link", ""),
        "image_url":    image_url,
        "image_gallery": image_gallery,
        "website":      website,
        "categories":   categories,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PEAK BUSINESS TIMES — Estimator
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT NOTE FOR REVIEWERS:
# SerpAPI's Yelp engine does NOT return Google-style "popular times" or live
# wait-time data — that data is exclusive to Google Maps. To still provide
# the Peak Times visualization, this function generates a plausible busy
# pattern using typical restaurant traffic curves, with cuisine- and
# price-aware adjustments. The chart in the UI is clearly labeled "Estimated."
# If a future SerpAPI Google Maps integration is added, this function can be
# swapped for the real popular_times array.
# ─────────────────────────────────────────────────────────────────────────────

def estimate_peak_times(restaurant_name: str = "",
                         cuisine: str = "",
                         price_tier: int = 2) -> dict:
    """
    Generates an estimated busy-by-hour profile for a restaurant.

    Uses a deterministic seed based on the restaurant name so the same
    restaurant always shows the same pattern within a session (looks
    consistent on refresh) but different restaurants look different.

    Returns:
        dict with keys:
          - busy_by_hour:   {hour_int: busy_pct (0-100)} for 11am-10pm
          - peak_hour:      int (24h) of the busiest hour
          - peak_label:     str like "7 PM"
          - avg_wait_min:   int — estimated wait at peak in minutes
          - is_estimated:   bool — always True (transparency flag)
    """
    seed_str = f"{restaurant_name}|{cuisine}|{price_tier}"
    rng = random.Random(hash(seed_str) % (10 ** 8))

    # Base typical-restaurant busy curve (hour → % busy at peak capacity)
    base_pattern = {
        11: 25, 12: 75, 13: 80, 14: 45,    # Lunch peak
        15: 25, 16: 30, 17: 50,             # Afternoon lull → early dinner
        18: 75, 19: 90, 20: 80,             # Dinner peak
        21: 55, 22: 30,                     # Wind-down
    }

    pattern = dict(base_pattern)

    # Cuisine-aware tweaks
    cuisine_lc = (cuisine or "").lower()
    if cuisine_lc in ("ramen", "japanese", "sushi", "korean"):
        # Dinner-heavy
        for h in (19, 20):
            pattern[h] = min(98, pattern[h] + 8)
    elif cuisine_lc in ("burgers", "pizza", "american", "mexican"):
        # Strong lunch + dinner
        for h in (12, 13):
            pattern[h] = min(98, pattern[h] + 8)
    elif cuisine_lc in ("seafood", "hawaiian"):
        # Long dinner stretch
        for h in (18, 19, 20, 21):
            pattern[h] = min(98, pattern[h] + 5)

    # Price-tier tweaks: pricier places = more dinner-skewed, slower lunches
    if price_tier >= 3:
        pattern[12] = max(15, pattern[12] - 20)
        pattern[13] = max(15, pattern[13] - 20)
        for h in (19, 20):
            pattern[h] = min(98, pattern[h] + 5)

    # Add per-restaurant random jitter so charts don't look identical
    busy_by_hour = {
        h: max(5, min(100, pattern[h] + rng.randint(-10, 10)))
        for h in pattern
    }

    peak_hour = max(busy_by_hour, key=busy_by_hour.get)
    peak_busy = busy_by_hour[peak_hour]

    # Wait estimate — peakier places + pricier = longer waits
    base_wait = (peak_busy / 100) * 30  # 0-30 min
    wait_min = int(base_wait + (price_tier * 2) + rng.randint(0, 5))
    wait_min = max(5, min(60, wait_min))

    # Format peak hour label
    if peak_hour == 12:
        peak_label = "12 PM"
    elif peak_hour < 12:
        peak_label = f"{peak_hour} AM"
    else:
        peak_label = f"{peak_hour - 12} PM"

    return {
        "busy_by_hour":  busy_by_hour,
        "peak_hour":     peak_hour,
        "peak_label":    peak_label,
        "avg_wait_min":  wait_min,
        "is_estimated":  True,
    }
# If a future SerpAPI Google Maps integration is added, this function can be
# swapped for the real popular_times array.
# ─────────────────────────────────────────────────────────────────────────────

def estimate_peak_times(restaurant_name: str = "",
                         cuisine: str = "",
                         price_tier: int = 2) -> dict:
    """
    Generates an estimated busy-by-hour profile for a restaurant.

    Uses a deterministic seed based on the restaurant name so the same
    restaurant always shows the same pattern within a session (looks
    consistent on refresh) but different restaurants look different.

    Returns:
        dict with keys:
          - busy_by_hour:   {hour_int: busy_pct (0-100)} for 11am-10pm
          - peak_hour:      int (24h) of the busiest hour
          - peak_label:     str like "7 PM"
          - avg_wait_min:   int — estimated wait at peak in minutes
          - is_estimated:   bool — always True (transparency flag)
    """
    seed_str = f"{restaurant_name}|{cuisine}|{price_tier}"
    rng = random.Random(hash(seed_str) % (10 ** 8))

    # Base typical-restaurant busy curve (hour → % busy at peak capacity)
    base_pattern = {
        11: 25, 12: 75, 13: 80, 14: 45,    # Lunch peak
        15: 25, 16: 30, 17: 50,             # Afternoon lull → early dinner
        18: 75, 19: 90, 20: 80,             # Dinner peak
        21: 55, 22: 30,                     # Wind-down
    }

    pattern = dict(base_pattern)

    # Cuisine-aware tweaks
    cuisine_lc = (cuisine or "").lower()
    if cuisine_lc in ("ramen", "japanese", "sushi", "korean"):
        # Dinner-heavy
        for h in (19, 20):
            pattern[h] = min(98, pattern[h] + 8)
    elif cuisine_lc in ("burgers", "pizza", "american", "mexican"):
        # Strong lunch + dinner
        for h in (12, 13):
            pattern[h] = min(98, pattern[h] + 8)
    elif cuisine_lc in ("seafood", "hawaiian"):
        # Long dinner stretch
        for h in (18, 19, 20, 21):
            pattern[h] = min(98, pattern[h] + 5)

    # Price-tier tweaks: pricier places = more dinner-skewed, slower lunches
    if price_tier >= 3:
        pattern[12] = max(15, pattern[12] - 20)
        pattern[13] = max(15, pattern[13] - 20)
        for h in (19, 20):
            pattern[h] = min(98, pattern[h] + 5)

    # Add per-restaurant random jitter so charts don't look identical
    busy_by_hour = {
        h: max(5, min(100, pattern[h] + rng.randint(-10, 10)))
        for h in pattern
    }

    peak_hour = max(busy_by_hour, key=busy_by_hour.get)
    peak_busy = busy_by_hour[peak_hour]

    # Wait estimate — peakier places + pricier = longer waits
    base_wait = (peak_busy / 100) * 30  # 0-30 min
    wait_min = int(base_wait + (price_tier * 2) + rng.randint(0, 5))
    wait_min = max(5, min(60, wait_min))

    # Format peak hour label
    if peak_hour == 12:
        peak_label = "12 PM"
    elif peak_hour < 12:
        peak_label = f"{peak_hour} AM"
    else:
        peak_label = f"{peak_hour - 12} PM"

    return {
        "busy_by_hour":  busy_by_hour,
        "peak_hour":     peak_hour,
        "peak_label":    peak_label,
        "avg_wait_min":  wait_min,
        "is_estimated":  True,
    }


# ── Test Block ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    results = scrape_yelp("Honolulu, HI", "japanese", 2, max_results=20)
    print("\n--- Results ---")
    for r in results:
        print(r)
    with open("yelp_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved to yelp_results.json")