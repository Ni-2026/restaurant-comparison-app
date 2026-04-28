"""
file_io.py
----------
Handles saving and reloading restaurant session data in CSV and JSON formats.

Main functions:
  - save_session()     → saves DataFrame to both CSV and JSON
  - load_session()     → reloads a saved session (auto-detects format)
  - export_to_csv()    → CSV-only export (convenience wrapper)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default directory for session files (relative to project root)
DEFAULT_SESSION_DIR = "sessions"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ensure_dir(directory: str) -> Path:
    """Creates the session directory if it doesn't exist, returns a Path object."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    """Returns a filesystem-safe timestamp string: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Raises ValueError if df is None or not a DataFrame."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame, got: " + str(type(df)))


# ── Save ───────────────────────────────────────────────────────────────────────

def save_session(
    df: pd.DataFrame,
    location: str = "",
    cuisine: str = "",
    budget: str = "",
    session_dir: str = DEFAULT_SESSION_DIR,
    filename_prefix: str = "session",
) -> dict[str, str]:
    """
    Saves the ranked results DataFrame as both a CSV and a JSON file.

    The JSON includes metadata (location, cuisine, budget, timestamp) plus the
    restaurant records so sessions are fully self-describing.

    File names follow the pattern:
        sessions/session_<timestamp>.csv
        sessions/session_<timestamp>.json

    Args:
        df:              Ranked DataFrame from data_pipeline.rank_restaurants().
        location:        Search location string (stored as metadata).
        cuisine:         Cuisine type string (stored as metadata).
        budget:          Budget/price level string (stored as metadata).
        session_dir:     Directory to write files into.
        filename_prefix: Prefix for file names (default "session").

    Returns:
        Dict with keys "csv" and "json" pointing to the saved file paths.

    Raises:
        ValueError:  If df is not a DataFrame.
        OSError:     If the directory cannot be created or files cannot be written.
    """
    _validate_dataframe(df)

    dir_path = _ensure_dir(session_dir)
    ts = _timestamp()
    base_name = f"{filename_prefix}_{ts}"

    csv_path  = dir_path / f"{base_name}.csv"
    json_path = dir_path / f"{base_name}.json"

    # ── Save CSV ──
    df.to_csv(csv_path, index=False)
    logger.info("Session saved to CSV: %s", csv_path)

    # ── Save JSON (with metadata wrapper) ──
    session_data = {
        "metadata": {
            "timestamp": ts,
            "location":  location,
            "cuisine":   cuisine,
            "budget":    budget,
            "count":     len(df),
        },
        "restaurants": df.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    logger.info("Session saved to JSON: %s", json_path)

    return {"csv": str(csv_path), "json": str(json_path)}


# ── Load ───────────────────────────────────────────────────────────────────────

def load_session(file_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Loads a previously saved session from either a CSV or JSON file.

    For CSV files, an empty metadata dict is returned (CSV has no metadata wrapper).
    For JSON files, the metadata dict is returned alongside the DataFrame.

    Args:
        file_path: Path to a .csv or .json session file.

    Returns:
        Tuple of (DataFrame, metadata_dict).
        The DataFrame has the same columns as when it was saved.
        metadata_dict contains: timestamp, location, cuisine, budget, count
        (empty dict for CSV files).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file extension is not .csv or .json.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Session file not found: '{file_path}'. "
            "Check the path or list available sessions with list_sessions()."
        )

    ext = path.suffix.lower()

    if ext == ".csv":
        df = _load_csv(path)
        metadata = {}

    elif ext == ".json":
        df, metadata = _load_json(path)

    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Use .csv or .json."
        )

    logger.info("Session loaded: %d restaurants from '%s'.", len(df), file_path)
    return df, metadata


def _load_csv(path: Path) -> pd.DataFrame:
    """Loads a CSV session file, coercing column types."""
    df = pd.read_csv(path)
    df = _coerce_types(df)
    return df


def _load_json(path: Path) -> tuple[pd.DataFrame, dict]:
    """Loads a JSON session file, returns (DataFrame, metadata)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both wrapped format {metadata, restaurants} and plain list
    if isinstance(data, dict) and "restaurants" in data:
        records  = data["restaurants"]
        metadata = data.get("metadata", {})
    elif isinstance(data, list):
        records  = data
        metadata = {}
    else:
        raise ValueError("Unrecognized JSON session format.")

    df = pd.DataFrame(records)
    df = _coerce_types(df)
    return df, metadata


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Re-applies correct dtypes after loading from flat file."""
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "reviews" in df.columns:
        df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce").fillna(0).astype(int)
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0).astype(int)
    if "price_level" in df.columns:
        df["price_level"] = pd.to_numeric(df["price_level"], errors="coerce").fillna(0).astype(int)
    return df


# ── CSV-Only Export ────────────────────────────────────────────────────────────

def export_to_csv(
    df: pd.DataFrame,
    file_path: str,
) -> str:
    """
    Exports the DataFrame to a CSV at the specified path.

    This is a lightweight convenience function for when you only need CSV
    (no JSON, no metadata) — for example, a one-click download from the Flask UI.

    Args:
        df:        DataFrame to export.
        file_path: Target file path (will create parent directories if needed).

    Returns:
        Absolute file path of the written CSV.

    Raises:
        ValueError: If df is not a DataFrame.
        OSError:    If the file cannot be written.
    """
    _validate_dataframe(df)
    path = Path(file_path)
    _ensure_dir(str(path.parent))
    df.to_csv(path, index=False)
    logger.info("CSV exported to: %s", path.resolve())
    return str(path.resolve())


# ── Session Discovery ──────────────────────────────────────────────────────────

def list_sessions(session_dir: str = DEFAULT_SESSION_DIR) -> list[dict]:
    """
    Returns a list of available session files in the sessions directory.

    Each entry is a dict with:
        path     : full file path string
        format   : "csv" or "json"
        filename : file name only
        modified : ISO-format last-modified timestamp

    Args:
        session_dir: Directory to scan.

    Returns:
        List of session info dicts, sorted newest-first.
        Returns an empty list if the directory doesn't exist.
    """
    dir_path = Path(session_dir)
    if not dir_path.exists():
        return []

    sessions = []
    for file in dir_path.iterdir():
        if file.suffix.lower() in (".csv", ".json"):
            sessions.append({
                "path":     str(file),
                "format":   file.suffix.lower().lstrip("."),
                "filename": file.name,
                "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
            })

    sessions.sort(key=lambda x: x["modified"], reverse=True)
    return sessions
