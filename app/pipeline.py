# pipeline.py
# Nizhen He — ITM352 Restaurant Comparison App
# Cleans and deduplicates raw scraped data using Pandas
#
# UPDATED: Now preserves image_url, website, and categories fields so they
# can be displayed in the results UI.

import re
import pandas as pd

PRICE_MAP = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}


def build_dataframe(raw_data: list[dict]) -> pd.DataFrame:
    """
    Takes raw scraped restaurant records and returns a clean,
    deduplicated Pandas DataFrame.
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

    # Step 2: Fill missing optional fields (including the new ones)
    df["review_count"] = df["review_count"].fillna(0)
    df["price"]        = df["price"].fillna("N/A")
    df["address"]      = df["address"].fillna("N/A")
    df["phone"]        = df["phone"].fillna("N/A")
    df["yelp_url"]     = df["yelp_url"].fillna("")

    # Newer optional fields — may not exist in older saved files,
    # so add them as empty columns first if missing.
    for col in ("image_url", "website", "categories"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")
    if "image_gallery" not in df.columns:
        df["image_gallery"] = df.apply(lambda row: [row.get("image_url", "")] if row.get("image_url") else [], axis=1)
    else:
        df["image_gallery"] = df["image_gallery"].fillna("").apply(lambda x: x if isinstance(x, list) else [])

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
    LAYER 1 — Yelp URL slug
    LAYER 2 — Normalized name
    LAYER 3 — Normalized name + address first line
    """
    before = len(df)
    df.sort_values("review_count", ascending=False, inplace=True)

    df["_url_key"] = df["yelp_url"].apply(_extract_url_slug)
    df.drop_duplicates(subset=["_url_key"], keep="first", inplace=True)
    after_l1 = len(df)

    df["_name_key"] = df["name"].apply(_normalize_name)
    df.drop_duplicates(subset=["_name_key"], keep="first", inplace=True)
    after_l2 = len(df)

    df["_addr_key"] = df["address"].apply(_normalize_address_line)
    df.drop_duplicates(subset=["_name_key", "_addr_key"], keep="first", inplace=True)
    after_l3 = len(df)

    print(f"[Dedup] Removed {before - after_l1} by URL  |  "
          f"{after_l1 - after_l2} by name  |  "
          f"{after_l2 - after_l3} by name+address")

    df.drop(columns=["_url_key", "_name_key", "_addr_key"], inplace=True)
    return df


# ── Normalization Helpers ─────────────────────────────────────────────────────

def _extract_url_slug(url: str) -> str:
    if not url:
        return ""
    match = re.search(r"/biz/([^?#]+)", url)
    return match.group(1).lower().strip() if match else url.lower().strip()


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _normalize_address_line(address: str) -> str:
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