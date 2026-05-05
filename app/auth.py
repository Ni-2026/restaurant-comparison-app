# auth.py
# Grace Kulhanek — ITM352 Restaurant Comparison App
#
# Handles user registration, login, session management, and profile persistence.
# Storage: flat JSON file (data/users.json) — no database required for this project.
# Passwords are hashed with werkzeug.security (bcrypt under the hood).
# Flask-Login manages the "stay logged in" session cookie via UserMixin.

import os
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

# ── Storage path ──────────────────────────────────────────────────────────────
# Placed in data/ so it's easy to find and back up separately from code.
USERS_FILE = os.path.join(os.path.dirname(__file__), "data", "users.json")


# ── Low-level file helpers ────────────────────────────────────────────────────

def _load_users() -> dict:
    """Reads users.json and returns a dict keyed by username.
    Returns empty dict if the file doesn't exist yet."""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict):
    """Writes the users dict back to disk atomically."""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


# ── Flask-Login User class ────────────────────────────────────────────────────
# Flask-Login needs is_authenticated, is_active, is_anonymous, get_id().
# UserMixin provides safe defaults; we only override what we need.

class User(UserMixin):
    def __init__(self, username: str, data: dict):
        self.id           = username          # Flask-Login uses get_id() → self.id
        self.username     = username
        self.email        = data.get("email", "")
        self.location     = data.get("location", "")    # default island/city
        self.budget       = data.get("budget", 2)       # 1–4, default $$
        self.restrictions = data.get("restrictions", []) # ["vegetarian", ...]
        self.created_at   = data.get("created_at", "")
        # history: list of past recommendation result dicts (capped at 20)
        self.history      = data.get("history", [])


# ── Public auth functions ─────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Creates a new user account after validating inputs.

    Returns:
        (True,  "")         — success
        (False, reason_str) — validation failure or duplicate username
    """
    username = username.strip()
    email    = email.strip()

    if not username or not email or not password:
        return False, "All fields are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()

    # Case-insensitive uniqueness check
    if username.lower() in {u.lower() for u in users}:
        return False, "That username is already taken."

    users[username] = {
        "email":         email,
        "password_hash": generate_password_hash(password),
        "location":      "",
        "budget":        2,
        "restrictions":  [],
        "created_at":    datetime.now().isoformat(),
        "history":       [],
    }
    _save_users(users)
    return True, ""


def login_user_lookup(username: str, password: str):
    """
    Validates credentials.

    Returns a User object on success, None on failure.
    The returned object is passed to flask_login.login_user().
    """
    users = _load_users()
    # Case-insensitive username lookup
    key = next((k for k in users if k.lower() == username.strip().lower()), None)
    if not key:
        return None
    if not check_password_hash(users[key]["password_hash"], password):
        return None
    return User(key, users[key])


def get_user(username: str):
    """
    Loads a User by username.
    Called by flask_login's @login_manager.user_loader on every request.
    """
    users = _load_users()
    if username not in users:
        return None
    return User(username, users[username])


def update_profile(username: str, location: str, budget: int,
                   restrictions: list) -> bool:
    """Saves updated preferences for a logged-in user. Returns True on success."""
    users = _load_users()
    if username not in users:
        return False
    users[username]["location"]     = location.strip()
    users[username]["budget"]       = int(budget)
    users[username]["restrictions"] = [r.strip().lower() for r in restrictions if r.strip()]
    _save_users(users)
    return True


def save_to_history(username: str, entry: dict):
    """
    Appends a search result summary to the user's history list.
    Keeps only the 20 most recent entries to prevent unbounded growth.

    entry should contain: { cuisine, location, budget, top_pick, score, saved_at }
    """
    users = _load_users()
    if username not in users:
        return
    users[username].setdefault("history", []).append(entry)
    users[username]["history"] = users[username]["history"][-20:]
    _save_users(users)
