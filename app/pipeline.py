# Nizhen He — ITM352 Restaurant Comparison App
# Date Updated - April 30, 2026
# Cleans and structures raw scraped data using Pandas

import pandas as pd


# ── Price Conversion Map ──────────────────────────────────────────────────────
# Maps Yelp's "$" symbols to numeric values for calculations
PRICE_MAP = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}


# ── Main Pipeline Function ────────────────────────────────────────────────────

def build_dataframe(raw_data: list[dict]) -> pd.DataFrame:
    """
    Takes raw scraped restaurant records and returns a clean Pandas DataFrame.

    Steps:
      1. Load list of dicts into DataFrame
      2. Drop rows missing critical fields (name, rating)
      3. Fill in sensible defaults for optional fields
      4. Convert types (rating → float, review_count → int, price → int)
      5. Remove duplicate restaurant names
      6. Reset index

    Args:
        raw_data (list[dict]): Output from scraper.scrape_yelp()

    Returns:
        pd.DataFrame: Clean restaurant data, or empty DataFrame if input is empty
    """
    if not raw_data:
        print("[Pipeline] No data to process.")
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)

    # ── Step 1: Drop rows with no name or no rating ────────────────────────────
    before = len(df)
    df.dropna(subset=["name", "rating"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[Pipeline] Dropped {dropped} rows missing name/rating.")

    if df.empty:
        return df

    # ── Step 2: Fill missing optional fields ──────────────────────────────────
    df["review_count"] = df["review_count"].fillna(0)
    df["price"]        = df["price"].fillna("N/A")
    df["address"]      = df["address"].fillna("N/A")
    df["phone"]        = df["phone"].fillna("N/A")
    df["yelp_url"]     = df["yelp_url"].fillna("")

    # ── Step 3: Type conversion ────────────────────────────────────────────────
    df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)

    # Convert price symbols → numeric (keeps "N/A" rows as 0 for scoring purposes)
    df["price_num"] = df["price"].map(PRICE_MAP).fillna(0).astype(int)

    # ── Step 4: Remove duplicate restaurant names (keep first occurrence) ──────
    df.drop_duplicates(subset=["name"], keep="first", inplace=True)

    # ── Step 5: Clean up index ─────────────────────────────────────────────────
    df.reset_index(drop=True, inplace=True)

    print(f"[Pipeline] Clean DataFrame: {len(df)} restaurants.")
    return df


# ── Summary Helper ────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> dict:
    """
    Returns basic summary statistics about the cleaned dataset.
    Useful for debugging and for passing context to the AI recommendation.

    Args:
        df (pd.DataFrame): Clean restaurant DataFrame

    Returns:
        dict with avg_rating, avg_reviews, price_distribution, total
    """
    if df.empty:
        return {}

    return {
        "total":              len(df),
        "avg_rating":         round(df["rating"].mean(), 2),
        "avg_reviews":        int(df["review_count"].mean()),
        "price_distribution": df["price"].value_counts().to_dict(),
    }