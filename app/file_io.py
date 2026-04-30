# Nizhen He — ITM352 Restaurant Comparison App
# Date Updated - April 30, 2026
# Saves and reloads comparison sessions as CSV and JSON

import os
import json
from datetime import datetime
import pandas as pd


# ── Output Directory ──────────────────────────────────────────────────────────
SESSIONS_DIR = "sessions"   # Folder where saved sessions are stored


def ensure_sessions_dir():
    """Creates the sessions/ folder if it doesn't already exist."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def generate_filename(location: str, cuisine: str, extension: str) -> str:
    """
    Generates a timestamped filename for a session file.

    Example output: "sessions/honolulu_japanese_2024-11-01_143022.csv"

    Args:
        location  (str): User's location input
        cuisine   (str): User's cuisine input
        extension (str): File extension ("csv" or "json")

    Returns:
        str: Full file path
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # Sanitize location/cuisine for use in filenames (no spaces or special chars)
    loc  = location.strip().lower().replace(" ", "_").replace(",", "")
    cui  = cuisine.strip().lower().replace(" ", "_")
    filename = f"{loc}_{cui}_{timestamp}.{extension}"
    return os.path.join(SESSIONS_DIR, filename)


# ── Save Functions ────────────────────────────────────────────────────────────

def save_to_csv(df: pd.DataFrame, location: str, cuisine: str) -> str:
    """
    Saves the scored restaurant DataFrame to a CSV file.

    Args:
        df       (pd.DataFrame): Scored and ranked restaurant data
        location (str): User location (used in filename)
        cuisine  (str): User cuisine (used in filename)

    Returns:
        str: Path to the saved file, or empty string if save failed
    """
    if df.empty:
        print("[FileIO] Cannot save: DataFrame is empty.")
        return ""

    ensure_sessions_dir()
    path = generate_filename(location, cuisine, "csv")

    try:
        df.to_csv(path, index=False)
        print(f"[FileIO] Saved CSV → {path}")
        return path
    except OSError as e:
        print(f"[FileIO] Error saving CSV: {e}")
        return ""


def save_to_json(df: pd.DataFrame, location: str, cuisine: str,
                 user_budget: int) -> str:
    """
    Saves the session as a JSON file, including user preferences and results.

    JSON structure:
    {
        "session": { "location", "cuisine", "budget", "saved_at" },
        "results": [ { ...restaurant fields... }, ... ]
    }

    Args:
        df          (pd.DataFrame): Scored and ranked restaurant data
        location    (str): User location
        cuisine     (str): User cuisine
        user_budget (int): User budget level (1–4)

    Returns:
        str: Path to the saved file, or empty string if save failed
    """
    if df.empty:
        print("[FileIO] Cannot save: DataFrame is empty.")
        return ""

    ensure_sessions_dir()
    path = generate_filename(location, cuisine, "json")

    session_data = {
        "session": {
            "location":  location,
            "cuisine":   cuisine,
            "budget":    user_budget,
            "saved_at":  datetime.now().isoformat(),
        },
        "results": df_to_records(df),
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        print(f"[FileIO] Saved JSON → {path}")
        return path
    except (OSError, TypeError) as e:
        print(f"[FileIO] Error saving JSON: {e}")
        return ""


# ── Load Functions ────────────────────────────────────────────────────────────

def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Reloads a previously saved CSV session into a DataFrame.

    Args:
        filepath (str): Path to the .csv session file

    Returns:
        pd.DataFrame: Restored restaurant data, or empty DataFrame if load failed
    """
    if not os.path.exists(filepath):
        print(f"[FileIO] File not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        print(f"[FileIO] Loaded CSV ← {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"[FileIO] Error loading CSV: {e}")
        return pd.DataFrame()


def load_from_json(filepath: str) -> tuple[dict, pd.DataFrame]:
    """
    Reloads a previously saved JSON session.

    Args:
        filepath (str): Path to the .json session file

    Returns:
        tuple: (session_meta dict, results DataFrame)
               Both are empty/empty if load fails.
    """
    if not os.path.exists(filepath):
        print(f"[FileIO] File not found: {filepath}")
        return {}, pd.DataFrame()

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        session_meta = data.get("session", {})
        results_list = data.get("results", [])
        df = pd.DataFrame(results_list) if results_list else pd.DataFrame()
        print(f"[FileIO] Loaded JSON ← {filepath} ({len(df)} rows)")
        return session_meta, df

    except (OSError, json.JSONDecodeError) as e:
        print(f"[FileIO] Error loading JSON: {e}")
        return {}, pd.DataFrame()


# ── Session Listing ───────────────────────────────────────────────────────────

def list_sessions() -> list[dict]:
    """
    Scans the sessions/ folder and returns metadata for all saved sessions.
    Used to populate a "reload session" dropdown in the UI.

    Returns:
        list[dict]: Each dict has { filename, path, extension, modified_at }
        Sorted newest first.
    """
    ensure_sessions_dir()
    sessions = []

    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith((".csv", ".json")):
            full_path = os.path.join(SESSIONS_DIR, fname)
            modified  = os.path.getmtime(full_path)
            sessions.append({
                "filename":    fname,
                "path":        full_path,
                "extension":   fname.rsplit(".", 1)[-1],
                "modified_at": datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M"),
            })

    return sorted(sessions, key=lambda x: x["modified_at"], reverse=True)


# ── Internal Helpers ──────────────────────────────────────────────────────────

def df_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Converts a DataFrame to a JSON-serializable list of dicts.
    Handles NaN values (which are not valid JSON) by replacing with None.

    Args:
        df (pd.DataFrame): Any restaurant DataFrame

    Returns:
        list[dict]: JSON-safe records
    """
    return df.where(pd.notna(df), other=None).to_dict(orient="records")