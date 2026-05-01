# scraper.py
# Nizhen He — ITM352 Restaurant Comparison App
# Fetches live Yelp data via SerpAPI (bypasses Yelp's DataDome CAPTCHA)

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

    Args:
        location   (str): e.g. "Honolulu, HI"
        cuisine    (str): e.g. "seafood"
        budget     (int): 1=$  2=$$  3=$$$  4=$$$$
        max_results(int): How many results to return (10 per page, paginates automatically)

    Returns:
        list[dict]: Each dict has keys:
            name, rating, review_count, price, address, phone, yelp_url
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
        # Only add price filter if budget specified — strict filtering can return 0 results
        if budget:
            params["attrs"] = f"RestaurantsPriceRange2:{budget}"

        print(f"[Scraper] Fetching results {start + 1}–{start + 10} "
              f"for '{cuisine}' in {location}...")

        try:
            search  = GoogleSearch(params)
            results = search.get_dict()

            # Check for API-level errors (bad key, quota exceeded, etc.)
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

            # Stop if Yelp returned fewer than 10 — end of results
            if len(businesses) < 10:
                print("[Scraper] Reached end of Yelp results.")
                break

            # Small delay between pages to be polite
            time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"[Scraper] Unexpected error: {e}")
            break

    print(f"[Scraper] Fetched {len(restaurants)} restaurants.")
    return restaurants[:max_results]


# ── Bulk Hawaii Scraper ───────────────────────────────────────────────────────

def scrape_hawaii_bulk(cuisine: str, budget: int,
                       per_location: int = 20) -> list[dict]:
    """
    Scrapes multiple Hawaii locations and returns one combined raw list.
    Pass the result to pipeline.build_dataframe() — it handles deduplication.

    Args:
        cuisine      (str): e.g. "seafood"
        budget       (int): 1–4
        per_location (int): Results per location (default 20)
                            9 locations × 20 = up to 180 raw → ~100 unique after dedup

    Returns:
        list[dict]: Combined raw records from all Hawaii locations
    """
    all_raw = []
    for location in HAWAII_LOCATIONS:
        raw = scrape_yelp(location, cuisine, budget, max_results=per_location)
        all_raw.extend(raw)
        time.sleep(random.uniform(1.0, 2.0))   # pause between locations
    print(f"[Scraper] Total raw results: {len(all_raw)}")
    return all_raw


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_result(biz: dict) -> dict:
    """
    Converts one SerpAPI Yelp organic result into the flat dict
    that pipeline.build_dataframe() expects.

    SerpAPI Yelp result keys used:
        title, rating, reviews, price, phone, link, neighborhoods
    """
    # Address: SerpAPI returns neighborhoods (e.g. "Chinatown") not a full address
    # We combine it with location for a readable label
    neighborhood = biz.get("neighborhoods", "")

    return {
        "name":         biz.get("title",    ""),
        "rating":       biz.get("rating",   None),
        "review_count": biz.get("reviews",  0),
        "price":        biz.get("price",    "N/A"),
        "address":      neighborhood if neighborhood else "N/A",
        "phone":        biz.get("phone",    "N/A"),
        "yelp_url":     biz.get("link",     ""),
    }


# ── Test Block ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: scrape 20 japanese restaurants in Honolulu, HI, budget=$$
    import json
    results = scrape_yelp("Honolulu, HI", "japanese", 2, max_results=20)
    print("\n--- Results ---")
    for r in results:
        print(r)
    # Save results to file
    with open("yelp_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved to yelp_results.json")