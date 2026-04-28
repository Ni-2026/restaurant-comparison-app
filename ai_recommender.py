"""
ai_recommender.py
-----------------
Generates a natural-language restaurant recommendation using the Claude API.

Main functions:
  - build_prompt()          → constructs the prompt from ranked results
  - call_claude_api()       → sends the prompt to Claude and returns raw response
  - parse_recommendation()  → extracts the text from the API response
  - get_recommendation()    → convenience wrapper that runs all three steps
"""

import logging
import os
from typing import Optional

import anthropic
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL         = "claude-sonnet-4-20250514"
MAX_TOKENS    = 500
TOP_N_FOR_AI  = 5   # Send only the top N restaurants to keep the prompt concise


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(
    df: pd.DataFrame,
    location: str,
    cuisine: str,
    budget: str,
) -> str:
    """
    Builds a prompt for Claude based on the top ranked restaurants.

    Args:
        df:       Ranked DataFrame from data_pipeline.rank_restaurants().
        location: User's search location string.
        cuisine:  Cuisine type searched.
        budget:   Price preference string.

    Returns:
        A formatted prompt string ready to send to Claude.
    """
    top = df.head(TOP_N_FOR_AI)

    restaurant_list = ""
    for _, row in top.iterrows():
        restaurant_list += (
            f"\n{int(row.get('rank', 0))}. {row.get('name', 'Unknown')}"
            f" | Rating: {row.get('rating', 'N/A')}★"
            f" | Reviews: {row.get('reviews', 'N/A')}"
            f" | Price: {row.get('price', 'N/A')}"
            f" | Score: {row.get('score', 'N/A')}/100"
            f" | Category: {row.get('category', 'N/A')}"
        )

    prompt = f"""You are a friendly local food expert. A user is looking for {cuisine} restaurants 
in {location} with a budget of {budget or 'any price'}.

Here are the top ranked results based on ratings, reviews, and price fit:
{restaurant_list}

Please write a short, helpful recommendation paragraph (3-5 sentences) that:
- Highlights the top pick and why it stands out
- Mentions a good alternative option
- Feels warm and conversational, like advice from a friend who knows the area well
- Does NOT just repeat the numbers — give genuine insight based on the data

Keep it under 150 words."""

    return prompt


# ── API Call ───────────────────────────────────────────────────────────────────

def call_claude_api(
    prompt: str,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Sends a prompt to the Claude API and returns the response text.

    Args:
        prompt:  The user prompt string.
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Returns:
        Response text string, or None if the call fails.

    Raises:
        EnvironmentError: If no API key is found.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "Anthropic API key not found.\n"
            "  Option 1: export ANTHROPIC_API_KEY='your_key_here'\n"
            "  Option 2: call_claude_api(prompt, api_key='your_key_here')\n"
            "  Get a key at https://console.anthropic.com"
        )

    try:
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_recommendation(message)

    except anthropic.AuthenticationError:
        logger.error("Invalid Anthropic API key.")
        return None
    except anthropic.RateLimitError:
        logger.error("Anthropic rate limit hit.")
        return None
    except anthropic.APIConnectionError:
        logger.error("Could not connect to Anthropic API. Check your internet connection.")
        return None
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return None


# ── Response Parser ────────────────────────────────────────────────────────────

def parse_recommendation(message) -> Optional[str]:
    """
    Extracts the text content from a Claude API response object.

    Args:
        message: The response object returned by the Anthropic client.

    Returns:
        The recommendation text string, or None if extraction fails.
    """
    try:
        for block in message.content:
            if block.type == "text":
                return block.text.strip()
    except Exception as exc:
        logger.error("Failed to parse Claude response: %s", exc)
    return None


# ── Convenience Wrapper ────────────────────────────────────────────────────────

def get_recommendation(
    df: pd.DataFrame,
    location: str,
    cuisine: str,
    budget: str = "",
    api_key: Optional[str] = None,
) -> str:
    """
    Runs the full recommendation pipeline in one call.

    Args:
        df:       Ranked DataFrame from data_pipeline.rank_restaurants().
        location: Search location string.
        cuisine:  Cuisine type string.
        budget:   Price preference string.
        api_key:  Anthropic API key (falls back to env var).

    Returns:
        Recommendation paragraph string.
        Returns a fallback message if the API call fails or df is empty.
    """
    if df is None or df.empty:
        return "No restaurant data available to generate a recommendation."

    try:
        prompt = build_prompt(df, location, cuisine, budget)
        recommendation = call_claude_api(prompt, api_key=api_key)

        if recommendation:
            return recommendation
        else:
            return (
                f"Based on the results, {df.iloc[0]['name']} is your top pick "
                f"with a score of {df.iloc[0]['score']}/100. "
                "(AI recommendation temporarily unavailable.)"
            )

    except Exception as exc:
        logger.error("get_recommendation failed: %s", exc)
        return "Could not generate a recommendation at this time."
