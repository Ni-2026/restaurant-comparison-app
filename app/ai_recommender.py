# ai_recommender.py
# Grace Kulhanek — ITM352 Restaurant Comparison App
#
# Handles all Claude API interaction:
#   build_prompt()        — crafts the system + user message
#   call_claude_api()     — POST to /v1/messages, handles all error cases
#   parse_recommendation()— validates and extracts the JSON response
#   get_recommendation()  — single call that wires all three together

import os
import json
import textwrap
import requests

CLAUDE_MODEL  = "claude-sonnet-4-20250514"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MAX_TOKENS    = 700


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(payload: dict) -> tuple[str, str]:
    """
    Constructs the system prompt and user message from the scored restaurant payload.

    The system prompt tells Claude to respond ONLY in strict JSON so we can
    reliably parse the recommendation inside the Flask route without fragile
    text parsing.

    Args:
        payload (dict): Output of scoring.to_recommendation_payload()

    Returns:
        (system_prompt, user_message)
    """
    prefs      = payload.get("user_preferences", {})
    budget_str = "$" * int(prefs.get("budget", 2))

    system_prompt = textwrap.dedent("""
        You are a helpful and knowledgeable Hawaii restaurant recommendation assistant.
        You will receive a JSON object with a user's dining preferences and a ranked
        list of restaurants with composite scores.

        Respond ONLY with a valid JSON object. No markdown code fences, no preamble,
        no text outside the JSON. Use exactly this structure:
        {
          "top_pick": "<restaurant name>",
          "reason": "<2-3 sentences: why this is the best match for the user's budget and taste>",
          "runner_up": "<second restaurant name>",
          "runner_up_reason": "<1-2 sentences>",
          "tip": "<one practical tip: best dish to order, reservation advice, parking, etc.>",
          "avoid": "<one lower-ranked restaurant to skip and a brief reason why>"
        }

        Base your reasoning on the composite score, user's budget, and the
        rating/review balance. Mention the restaurant name naturally in your reason.
        Keep the tone friendly and local — you know Hawaii well.
    """).strip()

    user_message = (
        f"I'm looking for {prefs.get('cuisine', 'restaurants')} restaurants "
        f"in {prefs.get('location', 'Hawaii')} "
        f"with a budget around {budget_str}.\n\n"
        f"Here are the top-ranked options with their composite scores:\n\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        f"Which restaurant do you recommend and why?"
    )
    return system_prompt, user_message


# ─────────────────────────────────────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_claude_api(system_prompt: str, user_message: str,
                    api_key: str = None) -> dict:
    """
    Sends the prompt to the Claude API and returns the parsed JSON response.

    Falls back to a mock recommendation if no API key is configured so the
    app never crashes during a demo or when the key isn't set.

    Args:
        system_prompt (str): Instruction context for Claude
        user_message  (str): The user's request with restaurant data
        api_key       (str): API key; reads ANTHROPIC_API_KEY env var if not passed

    Returns:
        dict with top_pick, reason, runner_up, runner_up_reason, tip, avoid
        OR {"error": "<message>"} on failure
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        # No key → show a mock so the app is still demonstrable
        print("[AI] No API key — using mock recommendation.")
        return _mock_recommendation()

    try:
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      CLAUDE_MODEL,
                "max_tokens": MAX_TOKENS,
                "system":     system_prompt,
                "messages":   [{"role": "user", "content": user_message}],
            },
            timeout=30,
        )
        resp.raise_for_status()

    except requests.exceptions.Timeout:
        return {"error": "Claude API timed out. Please try again."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"Claude API HTTP error: {e}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Could not reach Claude API: {e}"}

    return parse_recommendation(resp.json())


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_recommendation(raw: dict) -> dict:
    """
    Extracts and validates the JSON recommendation from the raw Claude response.

    Handles the edge case where Claude wraps its JSON in markdown code fences
    despite the system prompt saying not to.

    Returns dict with recommendation keys, or {"error": "..."} on failure.
    """
    try:
        text = raw["content"][0]["text"].strip()

        # Strip markdown code fences if present (defensive)
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

        rec = json.loads(text)

        required = ["top_pick", "reason", "runner_up", "runner_up_reason", "tip"]
        missing  = [k for k in required if k not in rec]
        if missing:
            return {"error": f"AI response missing fields: {missing}"}

        return rec

    except (KeyError, IndexError):
        return {"error": "Unexpected API response structure."}
    except json.JSONDecodeError as e:
        return {"error": f"Could not parse AI JSON response: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY — called from app.py
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendation(df, user_budget: int, location: str,
                        cuisine: str, api_key: str = None) -> dict:
    """
    End-to-end: scores DataFrame → builds prompt → calls Claude → returns result.

    Args:
        df          (pd.DataFrame): Scored and ranked restaurant DataFrame
        user_budget (int):          User's budget level (1–4)
        location    (str):          User's location/island input
        cuisine     (str):          User's cuisine input
        api_key     (str):          Optional API key override

    Returns:
        dict with top_pick, reason, runner_up, runner_up_reason, tip, avoid
    """
    from scoring import to_recommendation_payload   # local import avoids circular deps
    payload      = to_recommendation_payload(df, user_budget, location, cuisine)
    system, user = build_prompt(payload)
    return call_claude_api(system, user, api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK — shown when no API key is set
# ─────────────────────────────────────────────────────────────────────────────

def _mock_recommendation() -> dict:
    return {
        "top_pick":         "Top Result",
        "reason":           (
            "Based on the composite scoring — balancing rating, review volume, "
            "and price fit — this restaurant stands out as the best match for "
            "your budget and preferences. It has consistently strong reviews "
            "from locals and visitors alike."
        ),
        "runner_up":        "Second Result",
        "runner_up_reason": (
            "A close second with excellent ratings and a price point that fits "
            "your budget well."
        ),
        "tip":  "Check their Yelp page for current hours and to see recent photos before you go.",
        "avoid": "The lowest-ranked option in these results has fewer reviews, "
                 "which makes its score less reliable.",
        "_mock": True,
    }
