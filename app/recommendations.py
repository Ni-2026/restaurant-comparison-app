# recommendations.py
# Grace Kulhanek — ITM352 Restaurant Comparison App
#
# Generates personalized "For You" picks based on the logged-in user's
# search history. Pure analysis — no external API calls — so this is fast
# and free to call on every page visit.

import os
import json
import random
from collections import Counter
from datetime import datetime

from pipeline import build_dataframe
from scoring import score_restaurants


def analyze_user_preferences(user) -> dict:
    """Counts cuisines, locations, and budgets across user.history."""
    if not user or not getattr(user, "history", None):
        return {
            "top_cuisines": [],
            "top_locations": [],
            "avg_budget": getattr(user, "budget", 2) if user else 2,
            "total_searches": 0,
            "cuisine_counts": {},
        }

    cuisines = [h.get("cuisine") for h in user.history if h.get("cuisine")]
    locations = [h.get("location") for h in user.history if h.get("location")]
    budgets = [h.get("budget") for h in user.history if h.get("budget")]

    cuisine_counts = Counter(cuisines)
    location_counts = Counter(locations)
    avg_budget = round(sum(budgets) / len(budgets)) if budgets else 2

    return {
        "top_cuisines": [c for c, _ in cuisine_counts.most_common(3)],
        "top_locations": [l for l, _ in location_counts.most_common(2)],
        "avg_budget": avg_budget,
        "total_searches": len(user.history),
        "cuisine_counts": dict(cuisine_counts),
    }


def generate_for_you_picks(
    user, demo_path: str = None, max_picks: int = 5, shuffle_seed: int = None
) -> dict:
    """
    Generates personalized restaurant picks based on user's search history.

    Args:
        user: Flask-Login User object (must have .history)
        demo_path: Path to yelp_results.json
        max_picks: Number of recommendations to return
        shuffle_seed: Seed for deterministic shuffling

    Returns:
        dict with picks, based_on, preferences, generated_at, and optional error
    """
    prefs = analyze_user_preferences(user)

    # If the user is brand new with no history
    if prefs["total_searches"] == 0:
        return {
            "picks": [],
            "based_on": "",
            "preferences": prefs,
            "error": "Run a few searches first — once you do, we'll spot patterns and recommend new spots you'll love.",
            "generated_at": "",
        }

    cuisine = (
        prefs["top_cuisines"][0] if prefs["top_cuisines"] else "japanese"
    )
    location = (
        prefs["top_locations"][0]
        if prefs["top_locations"]
        else (getattr(user, "location", "") or "Honolulu, HI")
    )
    budget = prefs["avg_budget"]

    # Load candidate pool from local demo data
    raw = _load_demo_pool(demo_path)
    if not raw:
        return {
            "picks": [],
            "based_on": cuisine,
            "preferences": prefs,
            "error": "Restaurant pool unavailable — try again later.",
            "generated_at": "",
        }

    # Build & score
    df = build_dataframe(raw)
    if df.empty:
        return {
            "picks": [],
            "based_on": cuisine,
            "preferences": prefs,
            "error": "No valid candidate restaurants right now.",
            "generated_at": "",
        }

    df_ranked = score_restaurants(df, user_budget=budget)

    # Filter out already-recommended top picks for variety
    seen_top_picks = {h.get("top_pick") for h in user.history if h.get("top_pick")}
    fresh = df_ranked[~df_ranked["name"].isin(seen_top_picks)]

    # Fall back to full ranked list if not enough fresh ones
    pool = fresh if len(fresh) >= max_picks else df_ranked

    # Take the top 2× max_picks then shuffle for variety
    seed = (
        shuffle_seed if shuffle_seed is not None 
        else int(datetime.now().timestamp())
    )
    rng = random.Random(seed)

    pool_size = min(len(pool), max_picks * 2)
    top_pool = pool.head(pool_size).to_dict(orient="records")
    rng.shuffle(top_pool)

    chosen = top_pool[:max_picks]
    # Re-sort the chosen picks by score so the strongest pick is on top
    chosen.sort(key=lambda r: r.get("score", 0), reverse=True)

    picks = [_pack_pick(row, prefs, cuisine) for row in chosen]

    return {
        "picks": picks,
        "based_on": cuisine,
        "preferences": prefs,
        "generated_at": datetime.now().strftime("%I:%M %p"),
    }


def _load_demo_pool(demo_path: str = None) -> list:
    """Loads the local yelp_results.json."""
    if demo_path is None:
        demo_path = os.path.join(os.path.dirname(__file__), "yelp_results.json")
    if not os.path.exists(demo_path):
        return []
    try:
        with open(demo_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[Recommender] Could not load demo pool: {e}")
        return []


def _pack_pick(row: dict, prefs: dict, primary_cuisine: str) -> dict:
    """Turns a single DataFrame row into the dict the For You template uses."""
    return {
        "name": row.get("name", ""),
        "rating": float(row.get("rating") or 0),
        "review_count": int(row.get("review_count") or 0),
        "price": row.get("price", "N/A"),
        "address": row.get("address", "N/A"),
        "phone": row.get("phone", "N/A"),
        "yelp_url": row.get("yelp_url", ""),
        "image_url": row.get("image_url", ""),
        "website": row.get("website", ""),
        "score": round(float(row.get("score") or 0), 3),
        "match_reason": _build_match_reason(row, prefs, primary_cuisine),
    }


def _build_match_reason(row: dict, prefs: dict, cuisine: str) -> str:
    """One short sentence explaining why this pick fits the user."""
    reasons = []

    cnt = prefs.get("cuisine_counts", {}).get(cuisine, 0)
    if cnt:
        s = "search" if cnt == 1 else "searches"
        reasons.append(f"You've made {cnt} {cuisine.title()} {s}")

    rating = float(row.get("rating") or 0)
    if rating >= 4.5:
        reasons.append(f"highly rated ({rating:.1f}★)")
    elif rating >= 4.2:
        reasons.append(f"well-reviewed ({rating:.1f}★)")

    if int(row.get("review_count") or 0) >= 1000:
        reasons.append("popular with locals")

    return " • ".join(reasons) if reasons else "A great find for you"
