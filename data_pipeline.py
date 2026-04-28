"""
data_pipeline.py
----------------
Cleans raw scraped data and produces a ranked DataFrame of restaurants.

Main functions:
  - clean_data()                → normalizes and fills missing values
  - calculate_composite_score() → weighted scoring across rating, reviews, price
  - rank_restaurants()          → sorts by composite score, returns final DataFrame
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Scoring Weights ────────────────────────────────────────────────────────────
# Weights must sum to 1.0.  Adjust here to re-balance the ranking formula.
SCORE_WEIGHTS = {
    "rating":  0.50,   # Star rating (1–5) contributes most
    "reviews": 0.30,   # Log-normalized review count rewards popularity
    "price":   0.20,   # Price fit vs. user's budget preference
}

# Maps Yelp price symbols → integer levels
PRICE_MAP = {
    "$":    1,
    "$$":   2,
    "$$$":  3,
    "$$$$": 4,
    "N/A":  0,        # treated as unknown → neutral score
}

# Inverse map for converting budget string input to numeric
BUDGET_NUMERIC = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean_data(restaurants: list[dict]) -> pd.DataFrame:
    """
    Converts raw restaurant dicts into a clean, typed DataFrame.

    Steps:
      1. Build DataFrame from list of dicts.
      2. Drop exact duplicates.
      3. Coerce rating → float; fill NaN with column median.
      4. Coerce reviews → int; fill NaN with 0.
      5. Map price string → int (PRICE_MAP); unknown → 0.
      6. Strip whitespace from string columns.

    Args:
        restaurants: List of dicts with keys: name, rating, reviews, price,
                     category, url.  Extra keys are preserved.

    Returns:
        Cleaned pandas DataFrame.  An empty DataFrame is returned (with correct
        columns) when the input list is empty.

    Raises:
        TypeError: If restaurants is not a list.
    """
    if not isinstance(restaurants, list):
        raise TypeError(f"Expected a list, got {type(restaurants).__name__}.")

    expected_cols = ["name", "rating", "reviews", "price", "category", "url"]

    if not restaurants:
        logger.warning("clean_data received an empty list; returning empty DataFrame.")
        return pd.DataFrame(columns=expected_cols + ["price_level"])

    df = pd.DataFrame(restaurants)

    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Drop fully duplicate rows
    initial_len = len(df)
    df = df.drop_duplicates(subset=["name"])
    dropped = initial_len - len(df)
    if dropped:
        logger.info("Dropped %d duplicate entries.", dropped)

    # ── rating ──
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    median_rating = df["rating"].median()
    if pd.isna(median_rating):
        median_rating = 0.0
    df["rating"] = df["rating"].fillna(median_rating).clip(0, 5)

    # ── reviews ──
    df["reviews"] = (
        pd.to_numeric(df["reviews"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )

    # ── price → price_level (int) ──
    df["price"] = df["price"].fillna("N/A").astype(str).str.strip()
    df["price_level"] = df["price"].map(PRICE_MAP).fillna(0).astype(int)

    # ── strip string columns ──
    for col in ["name", "category", "url"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    logger.info("clean_data finished: %d restaurants.", len(df))
    return df


# ── Scoring ────────────────────────────────────────────────────────────────────

def _normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1].  All-equal values → 0.5."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def _price_fit_score(price_level: pd.Series, target: int) -> pd.Series:
    """
    Scores each restaurant's price against the user's target budget.

    Exact match = 1.0; each step away loses 0.25 (minimum 0.0).
    Unknown price (0) → neutral 0.5.
    """
    scores = price_level.apply(
        lambda p: 0.5 if p == 0 else max(0.0, 1.0 - 0.25 * abs(p - target))
    )
    return scores


def calculate_composite_score(
    df: pd.DataFrame,
    budget: str = "$$",
    weights: dict = None,
) -> pd.DataFrame:
    """
    Adds a 'score' column (0–100) to the DataFrame using a weighted formula.

    Score components:
      • rating_norm   : min-max normalized star rating
      • reviews_norm  : min-max normalized log(reviews + 1)
      • price_fit     : how well the price matches the user's budget

    Args:
        df:      Cleaned DataFrame from clean_data().
        budget:  User's desired price level string ("$" to "$$$$").
        weights: Optional dict overriding SCORE_WEIGHTS.
                 Must contain keys: rating, reviews, price.

    Returns:
        DataFrame with an added 'score' column (float, 0–100).

    Raises:
        ValueError: If df is missing required columns.
        KeyError:   If weights dict is missing required keys.
    """
    required = {"rating", "reviews", "price_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    if df.empty:
        df["score"] = pd.Series(dtype=float)
        return df

    w = weights if weights is not None else SCORE_WEIGHTS
    for key in ("rating", "reviews", "price"):
        if key not in w:
            raise KeyError(f"Weights dict missing key: '{key}'")

    target_price = BUDGET_NUMERIC.get(budget, 2)   # default $$

    rating_norm  = _normalize_series(df["rating"])
    reviews_norm = _normalize_series(np.log1p(df["reviews"]))
    price_fit    = _price_fit_score(df["price_level"], target_price)

    composite = (
        w["rating"]  * rating_norm  +
        w["reviews"] * reviews_norm +
        w["price"]   * price_fit
    )

    df = df.copy()
    df["score"] = (composite * 100).round(2)
    logger.info("Scores calculated. Range: %.1f – %.1f", df["score"].min(), df["score"].max())
    return df


# ── Ranking ────────────────────────────────────────────────────────────────────

def rank_restaurants(
    df: pd.DataFrame,
    top_n: int = None,
) -> pd.DataFrame:
    """
    Sorts restaurants by composite score (descending) and adds a 'rank' column.

    Args:
        df:    DataFrame with a 'score' column (output of calculate_composite_score).
        top_n: If provided, returns only the top N rows.

    Returns:
        Sorted DataFrame with an integer 'rank' column starting at 1.
        Columns returned: rank, name, rating, reviews, price, score, category, url.

    Raises:
        ValueError: If 'score' column is absent.
    """
    if "score" not in df.columns:
        raise ValueError("'score' column not found. Run calculate_composite_score first.")

    ranked = df.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    if top_n is not None:
        ranked = ranked.head(top_n)

    # Reorder columns for clean display
    display_cols = ["rank", "name", "rating", "reviews", "price", "score", "category", "url"]
    existing = [c for c in display_cols if c in ranked.columns]
    extra    = [c for c in ranked.columns if c not in display_cols]
    ranked   = ranked[existing + extra]

    logger.info("rank_restaurants: returning %d rows.", len(ranked))
    return ranked


# ── Convenience Wrapper ────────────────────────────────────────────────────────

def run_pipeline(
    restaurants: list[dict],
    budget: str = "$$",
    max_results: int = 10,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Runs the full pipeline in one call:
        clean_data → calculate_composite_score → rank_restaurants

    Args:
        restaurants: Raw list of dicts from scraper.py.
        budget:      User's price preference ("$" – "$$$$").
        max_results: Maximum number of ranked results to return.
        weights:     Optional score weight overrides.

    Returns:
        Final ranked DataFrame ready for display or export.
    """
    df = clean_data(restaurants)
    if df.empty:
        logger.warning("Pipeline received no data; returning empty DataFrame.")
        return df
    df = calculate_composite_score(df, budget=budget, weights=weights)
    df = rank_restaurants(df, top_n=max_results)
    return df
