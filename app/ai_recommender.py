# ai_recommender.py
# Grace Kulhanek — ITM352 Restaurant Comparison App
#
# Provides AI-powered restaurant recommendations using Claude API.
# Falls back to mock recommendations if API is unavailable.

import os
import json
import textwrap
import requests


def get_recommendation(df_ranked, user_budget, location, cuisine):
    """
    Analyzes the top restaurants from the ranked DataFrame and returns
    an AI-powered recommendation with explanation.
    
    Returns a dict with:
      - top_pick (str): name of recommended restaurant
      - runner_up (str): name of second choice
      - why (str): explanation
      - _mock (bool): True if this is a mock recommendation (API unavailable)
    """
    
    if df_ranked.empty:
        return {
            "top_pick": "No restaurants available",
            "runner_up": "",
            "why": "Unable to generate recommendation: no data",
            "_mock": True,
        }
    
    try:
        # Try to use Claude API if available
        return _get_claude_recommendation(df_ranked, user_budget, location, cuisine)
    except Exception as e:
        print(f"[AI] Claude API failed: {e}. Using mock recommendation.")
        return _get_mock_recommendation(df_ranked, user_budget, location, cuisine)


def _get_claude_recommendation(df_ranked, user_budget, location, cuisine):
    """Attempts to get recommendation from Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    # Prepare top 5 restaurants for Claude
    top_5 = df_ranked.head(5).to_dict(orient="records")
    
    prompt = textwrap.dedent(f"""
    The user is looking for {cuisine} restaurants in {location} with a budget of ${user_budget * 10}-${user_budget * 40}.
    
    Here are the top 5 restaurants ranked by our scoring algorithm:
    {json.dumps(top_5, indent=2)}
    
    Based on ratings, price, reviews, and location, recommend the single best restaurant for this user.
    Respond with a JSON object:
    {{
        "top_pick": "restaurant name",
        "runner_up": "second choice name",
        "why": "Brief explanation (1-2 sentences) why this is the best choice"
    }}
    """).strip()
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=10,
    )
    
    response.raise_for_status()
    result = response.json()
    text = result["content"][0]["text"]
    
    # Parse JSON from response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON in Claude response")
    
    rec = json.loads(text[start:end])
    rec["_mock"] = False
    # Rename 'why' to 'reason' for template compatibility
    if "why" in rec:
        rec["reason"] = rec.pop("why")
    if "why_runner_up" in rec:
        rec["runner_up_reason"] = rec.pop("why_runner_up")
    return rec


def _get_mock_recommendation(df_ranked, user_budget, location, cuisine):
    """
    Returns a mock recommendation based on the ranked DataFrame.
    Used when Claude API is unavailable.
    """
    top_1 = df_ranked.iloc[0] if len(df_ranked) > 0 else None
    top_2 = df_ranked.iloc[1] if len(df_ranked) > 1 else None
    
    if top_1 is None:
        return {
            "top_pick": "No recommendation available",
            "runner_up": "",
            "reason": "No restaurants found.",
            "tip": "",
            "_mock": True,
        }
    
    return {
        "top_pick": top_1.get("name", "Unknown"),
        "runner_up": top_2.get("name", "") if top_2 is not None else "",
        "reason": f"Based on ratings, reviews, and your ${user_budget * 10}-${user_budget * 40} budget, this is the best match for {cuisine} in {location}.",
        "tip": f"Phone: {top_1.get('phone', 'N/A')} | Address: {top_1.get('address', 'N/A')}",
        "runner_up_reason": "",
        "_mock": True,
    }
