# pipeline.py
# Nizhen He — ITM352 Restaurant Comparison App
# Cleans and deduplicates raw scraped data using Pandas

import re
import pandas as pd

PRICE_MAP = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}


def build_dataframe(raw_data: list[dict]) -> pd.DataFrame:
    """
    Takes raw scraped restaurant records (can be combined from multiple locations)
    and returns a clean, deduplicated Pandas DataFrame.

    Steps:
      1. Load into DataFrame
      2. Drop rows missing name or rating
      3. Fill defaults for optional fields
      4. Convert types
      5. THREE-LAYER deduplication (URL slug → normalized name → name+address)
      6. Reset index
    """
    if not raw_data:
        print("[Pipeline] No data to process.")
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)
    start_count = len(df)

    # Step 1: Drop rows with no name or no rating
    df.dropna(subset=["name", "rating"], inplace=True)
    if df.empty:
        return df

    # Step 2: Fill missing optional fields
    df["review_count"] = df["review_count"].fillna(0)
    df["price"]        = df["price"].fillna("N/A")
    df["address"]      = df["address"].fillna("N/A")
    df["phone"]        = df["phone"].fillna("N/A")
    df["yelp_url"]     = df["yelp_url"].fillna("")

    # Step 3: Type conversion
    df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
    df["price_num"]    = df["price"].map(PRICE_MAP).fillna(0).astype(int)

    # Step 4: Three-layer deduplication
    df = deduplicate(df)

    # Step 5: Reset index
    df.reset_index(drop=True, inplace=True)

    removed = start_count - len(df)
    print(f"[Pipeline] {len(df)} unique restaurants ({removed} duplicates removed).")
    return df


# ── Three-Layer Deduplication ─────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate restaurants using three layers, from most to least reliable.

    LAYER 1 — Yelp URL slug
        The /biz/restaurant-name-city part of the URL is unique per restaurant.
        Two rows with the same slug are always the same place.
        When duplicates exist, keeps the row with MORE reviews (fresher data).

        Catches: same restaurant scraped from both "Honolulu, HI" and "Waikiki, HI"

    LAYER 2 — Normalized name
        Lowercase + strip punctuation + collapse spaces, then dedupe.
        Catches: "Mama's Fish House" vs "mamas fish house" vs "Mama's Fish House "

    LAYER 3 — Normalized name + address first line
        Only removes a row when BOTH the normalized name AND street address match.
        Safe for chains: "McDonald's, 123 King St" and "McDonald's, 456 Ala Moana"
        are kept as two separate entries.

        Catches: exact same location scraped twice with slightly different metadata.
    """
    before = len(df)

    # When there are duplicates, keep the one with the most reviews
    df.sort_values("review_count", ascending=False, inplace=True)

    # Layer 1: Yelp URL slug
    df["_url_key"] = df["yelp_url"].apply(_extract_url_slug)
    df.drop_duplicates(subset=["_url_key"], keep="first", inplace=True)
    after_l1 = len(df)

    # Layer 2: Normalized name
    df["_name_key"] = df["name"].apply(_normalize_name)
    df.drop_duplicates(subset=["_name_key"], keep="first", inplace=True)
    after_l2 = len(df)

    # Layer 3: Normalized name + address first line
    df["_addr_key"] = df["address"].apply(_normalize_address_line)
    df.drop_duplicates(subset=["_name_key", "_addr_key"], keep="first", inplace=True)
    after_l3 = len(df)

    print(f"[Dedup] Removed {before - after_l1} by URL  |  "
          f"{after_l1 - after_l2} by name  |  "
          f"{after_l2 - after_l3} by name+address")

    # Drop helper columns — only needed internally
    df.drop(columns=["_url_key", "_name_key", "_addr_key"], inplace=True)
    return df


# ── Normalization Helpers ─────────────────────────────────────────────────────

def _extract_url_slug(url: str) -> str:
    """
    Extracts the stable /biz/slug from a Yelp URL, ignoring query parameters.

    "https://www.yelp.com/biz/mamas-fish-house-paia?osq=seafood"
    → "mamas-fish-house-paia"
    """
    if not url:
        return ""
    match = re.search(r"/biz/([^?#]+)", url)
    return match.group(1).lower().strip() if match else url.lower().strip()


def _normalize_name(name: str) -> str:
    """
    Lowercases, removes punctuation, and collapses whitespace.

    "Mama's Fish House " → "mamas fish house"
    "MAMAS FISH HOUSE"   → "mamas fish house"
    """
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)   # remove apostrophes, dashes, etc.
    name = re.sub(r"\s+", " ", name)      # collapse multiple spaces
    return name


def _normalize_address_line(address: str) -> str:
    """
    Extracts and normalizes just the first line (street number + name).
    Used in Layer 3 to distinguish chain locations.

    "799 Front St, Lahaina, HI 96761" → "799 front st"
    """
    if not isinstance(address, str) or address == "N/A":
        return ""
    first_line = address.split(",")[0].lower().strip()
    return re.sub(r"\s+", " ", first_line)


# ── Summary Helper ────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "total":              len(df),
        "avg_rating":         round(df["rating"].mean(), 2),
        "avg_reviews":        int(df["review_count"].mean()),
        "price_distribution": df["price"].value_counts().to_dict(),
    }