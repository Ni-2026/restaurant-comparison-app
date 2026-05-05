# test_backend.py
# Nizhen He — ITM352 Restaurant Comparison App
# Run this to verify each backend module is working correctly.
# Tests each layer independently so you can isolate where issues are.

import sys

print("=" * 55)
print("   Backend Test Suite — Restaurant Comparison App")
print("=" * 55)

passed = 0
failed = 0

def test(name, fn):
    """Runs a single test and reports pass/fail."""
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}")
        print(f"     → {e}")
        failed += 1


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA — used for pipeline, scoring, and file_io tests
# so you can test those without needing Yelp to be scraped first
# ─────────────────────────────────────────────────────────────────────────────
MOCK_RAW = [
    {"name": "Ramen House",    "rating": 4.5, "review_count": 320, "price": "$$",   "address": "123 Main St",  "phone": "808-111-1111", "yelp_url": "https://yelp.com/1"},
    {"name": "Sushi Palace",   "rating": 4.8, "review_count": 810, "price": "$$$",  "address": "456 Ocean Ave", "phone": "808-222-2222", "yelp_url": "https://yelp.com/2"},
    {"name": "Noodle Bar",     "rating": 3.9, "review_count": 95,  "price": "$",    "address": "789 King St",  "phone": "808-333-3333", "yelp_url": "https://yelp.com/3"},
    {"name": "Tokyo Bistro",   "rating": 4.2, "review_count": 430, "price": "$$",   "address": "321 Ala Moana","phone": "808-444-4444", "yelp_url": "https://yelp.com/4"},
    {"name": "Bad Entry",      "rating": None,"review_count": 0,   "price": None,   "address": None,           "phone": None,           "yelp_url": ""},  # Should be dropped
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ 1 ] pipeline.py")

from pipeline import build_dataframe, summarize

def test_pipeline_basic():
    df = build_dataframe(MOCK_RAW)
    assert not df.empty, "DataFrame should not be empty"
    assert "rating" in df.columns, "Missing 'rating' column"
    assert "review_count" in df.columns, "Missing 'review_count' column"
    assert "price_num" in df.columns, "Missing 'price_num' column"

def test_pipeline_drops_bad_rows():
    df = build_dataframe(MOCK_RAW)
    assert "Bad Entry" not in df["name"].values, "Row with None rating should be dropped"

def test_pipeline_empty_input():
    df = build_dataframe([])
    assert df.empty, "Empty input should return empty DataFrame"

def test_pipeline_types():
    df = build_dataframe(MOCK_RAW)
    assert df["rating"].dtype == float, "Rating should be float"
    assert df["review_count"].dtype == int, "review_count should be int"

def test_pipeline_summary():
    df = build_dataframe(MOCK_RAW)
    s = summarize(df)
    assert "avg_rating" in s, "Summary missing avg_rating"
    assert "total" in s, "Summary missing total"

test("DataFrame builds correctly",        test_pipeline_basic)
test("Drops rows missing name/rating",    test_pipeline_drops_bad_rows)
test("Handles empty input gracefully",    test_pipeline_empty_input)
test("Column types are correct",          test_pipeline_types)
test("summarize() returns stats",         test_pipeline_summary)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — scoring.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ 2 ] scoring.py")

from pipeline import build_dataframe
from scoring  import score_restaurants, get_top_n, to_recommendation_payload

df_clean = build_dataframe(MOCK_RAW)

def test_scoring_adds_columns():
    ranked = score_restaurants(df_clean, user_budget=2)
    for col in ["score", "rank", "rating_norm", "review_norm", "price_fit"]:
        assert col in ranked.columns, f"Missing column: {col}"

def test_scoring_rank_order():
    ranked = score_restaurants(df_clean, user_budget=2)
    scores = ranked["score"].tolist()
    assert scores == sorted(scores, reverse=True), "Restaurants should be sorted best → worst"

def test_scoring_rank_starts_at_1():
    ranked = score_restaurants(df_clean, user_budget=2)
    assert ranked["rank"].iloc[0] == 1, "Top result should have rank 1"

def test_scoring_score_range():
    ranked = score_restaurants(df_clean, user_budget=2)
    assert ranked["score"].between(0, 1).all(), "All scores should be between 0 and 1"

def test_scoring_empty_df():
    import pandas as pd
    result = score_restaurants(pd.DataFrame(), user_budget=2)
    assert result.empty, "Empty input should return empty DataFrame"

def test_top_n():
    ranked = score_restaurants(df_clean, user_budget=2)
    top3 = get_top_n(ranked, 3)
    assert len(top3) == 3, "get_top_n(3) should return 3 rows"
    assert top3["rank"].iloc[0] == 1, "First row should be rank 1"

def test_recommendation_payload():
    ranked  = score_restaurants(df_clean, user_budget=2)
    payload = to_recommendation_payload(ranked, 2, "Honolulu, HI", "japanese")
    assert "user_preferences" in payload, "Payload missing user_preferences"
    assert "top_restaurants"  in payload, "Payload missing top_restaurants"
    assert len(payload["top_restaurants"]) <= 5, "Should return at most 5 restaurants"

test("score_restaurants() adds required columns", test_scoring_adds_columns)
test("Results sorted best → worst",               test_scoring_rank_order)
test("Rank starts at 1",                          test_scoring_rank_starts_at_1)
test("All scores between 0 and 1",                test_scoring_score_range)
test("Handles empty DataFrame",                   test_scoring_empty_df)
test("get_top_n() returns correct count",         test_top_n)
test("to_recommendation_payload() structure",     test_recommendation_payload)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — file_io.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ 3 ] file_io.py")

import os
from file_io  import save_to_csv, save_to_json, load_from_csv, load_from_json, list_sessions
from pipeline import build_dataframe
from scoring  import score_restaurants

df_ranked = score_restaurants(build_dataframe(MOCK_RAW), user_budget=2)

def test_save_csv():
    path = save_to_csv(df_ranked, "Honolulu HI", "japanese")
    assert path != "", "save_to_csv should return a file path"
    assert os.path.exists(path), f"CSV file should exist at {path}"

def test_load_csv():
    path = save_to_csv(df_ranked, "Honolulu HI", "japanese")
    loaded = load_from_csv(path)
    assert not loaded.empty, "Loaded CSV should not be empty"
    assert "name" in loaded.columns, "Loaded CSV should have 'name' column"

def test_save_json():
    path = save_to_json(df_ranked, "Honolulu HI", "japanese", user_budget=2)
    assert path != "", "save_to_json should return a file path"
    assert os.path.exists(path), f"JSON file should exist at {path}"

def test_load_json():
    path = save_to_json(df_ranked, "Honolulu HI", "japanese", user_budget=2)
    meta, loaded = load_from_json(path)
    assert meta.get("location") == "Honolulu HI", "Meta should contain location"
    assert not loaded.empty, "Loaded JSON results should not be empty"

def test_load_missing_file():
    loaded = load_from_csv("sessions/does_not_exist.csv")
    assert loaded.empty, "Loading missing file should return empty DataFrame"

def test_list_sessions():
    sessions = list_sessions()
    assert isinstance(sessions, list), "list_sessions() should return a list"

test("save_to_csv() creates a file",          test_save_csv)
test("load_from_csv() restores data",         test_load_csv)
test("save_to_json() creates a file",         test_save_json)
test("load_from_json() restores data",        test_load_json)
test("load_from_csv() handles missing file",  test_load_missing_file)
test("list_sessions() returns a list",        test_list_sessions)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — scraper.py (SerpAPI-based, no live API call needed)
# Tests the parser and API key check without spending any of your 100 searches.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ 4 ] scraper.py  (parser & config — no live API call)")

from scraper import parse_result, API_KEY

# A mock SerpAPI organic result — matches the real structure SerpAPI returns
MOCK_SERP_RESULT = {
    "title":         "Nico's Pier 38",
    "rating":        4.5,
    "reviews":       1842,
    "price":         "$$",
    "phone":         "(808) 540-1377",
    "link":          "https://www.yelp.com/biz/nicos-pier-38-honolulu",
    "neighborhoods": "Iwilei",
}

# A result missing optional fields — scraper should handle gracefully
MOCK_SERP_MINIMAL = {
    "title":  "Mystery Cafe",
    "rating": 4.0,
    "reviews": 50,
}

def test_parse_result_full():
    result = parse_result(MOCK_SERP_RESULT)
    assert result["name"]         == "Nico's Pier 38",  "name should be title"
    assert result["rating"]       == 4.5,               "rating should match"
    assert result["review_count"] == 1842,              "review_count should map from reviews"
    assert result["price"]        == "$$",              "price should match"
    assert result["phone"]        == "(808) 540-1377",  "phone should match"
    assert "yelp.com" in result["yelp_url"],            "yelp_url should contain yelp.com"

def test_parse_result_maps_reviews_key():
    # SerpAPI uses "reviews" not "review_count" — parser must remap it
    result = parse_result(MOCK_SERP_RESULT)
    assert "review_count" in result, "output must have 'review_count' key for pipeline"

def test_parse_result_missing_fields():
    # Missing price/phone/neighborhoods should fall back to "N/A"
    result = parse_result(MOCK_SERP_MINIMAL)
    assert result["price"]   == "N/A", "missing price should default to N/A"
    assert result["phone"]   == "N/A", "missing phone should default to N/A"
    assert result["address"] == "N/A", "missing neighborhood should default to N/A"

def test_parse_result_output_keys():
    # All keys that pipeline.build_dataframe() expects must be present
    result = parse_result(MOCK_SERP_RESULT)
    required = ["name", "rating", "review_count", "price", "address", "phone", "yelp_url"]
    for key in required:
        assert key in result, f"parse_result() output missing required key: '{key}'"

def test_api_key_is_set():
    assert API_KEY != "PASTE_YOUR_SERPAPI_KEY_HERE", \
        "API key is still the placeholder — paste your SerpAPI key into scraper.py"

def test_parse_result_compatible_with_pipeline():
    # Full round-trip: parse → build_dataframe → should not be empty
    from pipeline import build_dataframe
    raw = [parse_result(MOCK_SERP_RESULT)]
    df  = build_dataframe(raw)
    assert not df.empty, "A parsed SerpAPI result should survive build_dataframe()"

test("parse_result() maps all fields correctly",      test_parse_result_full)
test("parse_result() remaps 'reviews' → 'review_count'", test_parse_result_maps_reviews_key)
test("parse_result() handles missing fields",         test_parse_result_missing_fields)
test("parse_result() returns all required keys",      test_parse_result_output_keys)
test("API key is set (not placeholder)",              test_api_key_is_set)
test("parse_result() output works with pipeline",     test_parse_result_compatible_with_pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
total = passed + failed
print()
print("=" * 55)
print(f"  Results: {passed}/{total} tests passed")
if failed == 0:
    print("  ✅ All tests passed — your backend is ready!")
else:
    print(f"  ⚠️  {failed} test(s) failed — see errors above.")
print("=" * 55)
print(f"\nPython version: {sys.version}")
