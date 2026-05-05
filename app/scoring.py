# Nizhen He — ITM352 Restaurant Comparison App
# Date Updated - April 30, 2026
# Composite scoring algorithm: ranks restaurants by rating, reviews, and price fit

import pandas as pd


# ── Scoring Weights ────────────────────────────────────────────────────────────
# These three weights must sum to 1.0
# Adjust them to change how much each factor influences the final rank.
WEIGHT_RATING  = 0.50   # Rating is the most important factor
WEIGHT_REVIEWS = 0.30   # More reviews = more trustworthy score
WEIGHT_PRICE   = 0.20   # How well the price matches the user's budget


# ── Main Scoring Function ─────────────────────────────────────────────────────

def score_restaurants(df: pd.DataFrame, user_budget: int) -> pd.DataFrame:
    """
    Adds a composite score to each restaurant and returns a ranked DataFrame.

    Scoring formula:
        score = (rating_norm * 0.50)
              + (review_norm * 0.30)
              + (price_fit   * 0.20)

    Where:
        rating_norm  = rating scaled 0–1 relative to all results
        review_norm  = log-scaled review count scaled 0–1
        price_fit    = 1 if price matches budget exactly,
                       falls off the further away the price is

    Args:
        df          (pd.DataFrame): Clean DataFrame from pipeline.build_dataframe()
        user_budget (int): User's budget choice (1–4)

    Returns:
        pd.DataFrame: Same DataFrame with new columns:
            rating_norm, review_norm, price_fit, score, rank
        Sorted by score descending (best first).
    """
    if df.empty:
        return df

    df = df.copy()  # Don't modify the original

    # ── Component 1: Normalized Rating (0–5 scale → 0–1) ─────────────────────
    df["rating_norm"] = normalize(df["rating"])

    # ── Component 2: Normalized Review Count (log scale) ──────────────────────
    # Log scale prevents a restaurant with 2000 reviews from completely
    # overshadowing one with 400 reviews — both are "well reviewed"
    import numpy as np
    log_reviews       = np.log1p(df["review_count"])  # log(1 + x) avoids log(0)
    df["review_norm"] = normalize(log_reviews)

    # ── Component 3: Price Fit (how close price is to user's budget) ──────────
    df["price_fit"] = df["price_num"].apply(
        lambda p: calculate_price_fit(p, user_budget)
    )

    # ── Composite Score ────────────────────────────────────────────────────────
    df["score"] = (
        df["rating_norm"]  * WEIGHT_RATING  +
        df["review_norm"]  * WEIGHT_REVIEWS +
        df["price_fit"]    * WEIGHT_PRICE
    ).round(4)

    # ── Rank (1 = best) ────────────────────────────────────────────────────────
    df.sort_values("score", ascending=False, inplace=True)
    df["rank"] = range(1, len(df) + 1)
    df.reset_index(drop=True, inplace=True)

    return df


# ── Helper Functions ──────────────────────────────────────────────────────────

def normalize(series: pd.Series) -> pd.Series:
    """
    Min-max normalizes a Series to the range [0, 1].
    If all values are equal, returns a Series of 1.0s.

    Args:
        series (pd.Series): Numeric column to normalize

    Returns:
        pd.Series: Values scaled between 0 and 1
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def calculate_price_fit(price_num: int, user_budget: int) -> float:
    """
    Returns a 0–1 score based on how close a restaurant's price is
    to the user's desired budget level.

    Distance 0 → 1.0 (perfect match)
    Distance 1 → 0.7
    Distance 2 → 0.3
    Distance 3 → 0.0

    Args:
        price_num   (int): Restaurant's numeric price (1–4), 0 if unknown
        user_budget (int): User's chosen budget (1–4)

    Returns:
        float: Price fit score between 0.0 and 1.0
    """
    if price_num == 0:
        return 0.5  # Unknown price gets a neutral score

    distance = abs(price_num - user_budget)
    fit_scores = {0: 1.0, 1: 0.7, 2: 0.3, 3: 0.0}
    return fit_scores.get(distance, 0.0)


def get_top_n(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Returns the top N restaurants by score.
    Used to pass a short list to the AI recommendation engine.

    Args:
        df (pd.DataFrame): Scored and ranked DataFrame
        n  (int): How many top results to return (default 5)

    Returns:
        pd.DataFrame: Top N rows
    """
    return df.head(n).copy()


def to_recommendation_payload(df: pd.DataFrame, user_budget: int,
                               location: str, cuisine: str) -> dict:
    """
    Packages the top-5 scored restaurants into a structured dict
    for the Claude API recommendation prompt (Grace's section).

    Args:
        df          (pd.DataFrame): Scored DataFrame
        user_budget (int): User's budget choice
        location    (str): User's location input
        cuisine     (str): User's cuisine input

    Returns:
        dict: JSON-serializable payload for AI recommendation
    """
    top5 = get_top_n(df, 5)

    restaurants_list = []
    for _, row in top5.iterrows():
        restaurants_list.append({
            "rank":         int(row["rank"]),
            "name":         row["name"],
            "rating":       float(row["rating"]) if pd.notna(row["rating"]) else None,
            "review_count": int(row["review_count"]),
            "price":        row["price"],
            "score":        float(row["score"]),
            "yelp_url":     row["yelp_url"],
        })

    return {
        "user_preferences": {
            "location": location,
            "cuisine":  cuisine,
            "budget":   user_budget,
        },
        "top_restaurants": restaurants_list,
    }
