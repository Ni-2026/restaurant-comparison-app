# app.py
# ITM352 — Hawaii Restaurant Comparison App
# Team: Nizhen He (Backend/Data), Grace Kulhanek (Frontend/AI), Sara Bautista (Viz/Testing)
#
# Flask entry point. Wires together:
#   scraper.py       — SerpAPI live Yelp data
#   pipeline.py      — cleaning + deduplication
#   scoring.py       — composite ranking
#   visualizations.py— bar chart + radar chart
#   ai_recommender.py— Claude API pick
#   auth.py          — user accounts + session history
#   file_io.py       — CSV/JSON session persistence
#
# Routes:
#   GET  /                 → home / search form
#   POST /search           → run full pipeline → redirect to results
#   GET  /results/<id>     → display results + charts + AI pick
#   GET  /login            → login page
#   POST /login            → process login
#   GET  /register         → register page
#   POST /register         → create account
#   GET  /logout           → log out
#   GET  /profile          → view/edit preferences
#   POST /profile          → save preferences
#   GET  /history          → past recommendations
#   GET  /chart/<id>       → serve bar chart PNG
#   GET  /radar/<id>       → serve radar chart HTML (for iframe)

import os
import sys
import json
import uuid
from datetime import datetime

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, send_file, abort)
from flask_login import (LoginManager, login_user as flask_login_user,
                         logout_user, login_required, current_user)

# ── Imports from our backend modules ─────────────────────────────────────────
from pipeline        import build_dataframe, summarize
from scoring         import score_restaurants, to_recommendation_payload
from visualizations  import generate_all_charts
from ai_recommender  import get_recommendation
from file_io         import save_to_csv, save_to_json
from scraper         import scrape_yelp, scrape_hawaii_bulk, HAWAII_LOCATIONS
from auth            import (register_user, login_user_lookup,
                              get_user, update_profile, save_to_history)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hawaii-eats-itm352-2026-dev-secret")

# ── Flask-Login ───────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view             = "login"
login_manager.login_message          = "Please log in to save your preferences."
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(user_id: str):
    """Called by Flask-Login on every request to reload the user from the cookie."""
    return get_user(user_id)

# ── In-memory result cache ────────────────────────────────────────────────────
# Stores result payloads by short UUID so the results page can be fetched by ID.
# Survives for the lifetime of the Flask process (sufficient for a class demo).
_cache: dict = {}

# ── Available cuisine types and islands ──────────────────────────────────────
CUISINES = [
    ("japanese",    "Japanese"),
    ("hawaiian",    "Hawaiian / Local"),
    ("seafood",     "Seafood"),
    ("american",    "American"),
    ("chinese",     "Chinese"),
    ("korean",      "Korean"),
    ("italian",     "Italian"),
    ("mexican",     "Mexican"),
    ("thai",        "Thai"),
    ("vietnamese",  "Vietnamese"),
    ("vegetarian",  "Vegetarian / Vegan"),
    ("pizza",       "Pizza"),
    ("burgers",     "Burgers"),
    ("ramen",       "Ramen"),
    ("sushi",       "Sushi"),
]

ISLANDS = [
    ("Honolulu, HI",         "Oahu — Honolulu"),
    ("Waikiki, Honolulu, HI","Oahu — Waikiki"),
    ("Kailua, HI",           "Oahu — Kailua"),
    ("Lahaina, Maui, HI",    "Maui — Lahaina"),
    ("Kihei, Maui, HI",      "Maui — Kihei"),
    ("Kailua-Kona, HI",      "Big Island — Kona"),
    ("Hilo, HI",             "Big Island — Hilo"),
    ("Lihue, Kauai, HI",     "Kauai — Lihue"),
]

RESTRICTIONS = ["vegetarian", "vegan", "gluten-free", "halal", "kosher", "nut-free"]


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template(
        "home.html",
        cuisines=CUISINES,
        islands=ISLANDS,
        restrictions=RESTRICTIONS,
        # Pre-fill from user profile if logged in
        user_location  =current_user.location     if current_user.is_authenticated else "",
        user_budget    =current_user.budget        if current_user.is_authenticated else 2,
        user_restrictions=current_user.restrictions if current_user.is_authenticated else [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH — main pipeline trigger
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/search", methods=["POST"])
def search():
    """
    Runs the full pipeline on form submission:
      1. Parse inputs
      2. Scrape Yelp via SerpAPI (or fall back to yelp_results.json for demo)
      3. build_dataframe() → clean + deduplicate
      4. score_restaurants() → rank
      5. generate charts
      6. get_recommendation() → Claude AI pick
      7. Cache results → redirect to /results/<id>
    """
    cuisine  = request.form.get("cuisine",  "japanese")
    location = request.form.get("location", "Honolulu, HI")
    budget   = int(request.form.get("budget", 2))
    max_r    = int(request.form.get("max_results", 20))
    mode     = request.form.get("mode", "live")     # "live" or "demo"
    restrictions = request.form.getlist("restrictions")  # checkbox values

    # ── Fetch raw data ────────────────────────────────────────────
    if mode == "demo":
        # Demo mode: load the pre-scraped sample data shipped in the repo
        sample_path = os.path.join(os.path.dirname(__file__), "yelp_results.json")
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            print(f"[App] Demo mode: loaded {len(raw)} records from yelp_results.json")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            flash(f"Could not load demo data: {e}", "error")
            return redirect(url_for("home"))
    else:
        # Live mode: call SerpAPI
        try:
            raw = scrape_yelp(location, cuisine, budget, max_results=max_r)
        except Exception as e:
            flash(f"Scrape failed: {e}. Try demo mode.", "error")
            return redirect(url_for("home"))

    if not raw:
        flash("No restaurants found for that search. Try different filters.", "warning")
        return redirect(url_for("home"))

    # ── Pipeline ──────────────────────────────────────────────────
    df = build_dataframe(raw)
    if df.empty:
        flash("No valid restaurant data after cleaning. Try different filters.", "warning")
        return redirect(url_for("home"))

    df_ranked = score_restaurants(df, user_budget=budget)
    stats     = summarize(df)

    # ── Charts ────────────────────────────────────────────────────
    session_id = str(uuid.uuid4())[:8]
    charts = generate_all_charts(
        df_ranked, session_id,
        top_n=min(5, len(df_ranked)),
        cuisine=cuisine,
        location=location,
    )

    # ── AI recommendation ─────────────────────────────────────────
    rec = get_recommendation(df_ranked, budget, location, cuisine)

    # ── Determine the single final pick ──────────────────────────
    # Prefer AI's top_pick if it matches a real result; otherwise rank-1
    ai_name      = rec.get("top_pick", "")
    pick_matches = df_ranked[df_ranked["name"] == ai_name]
    final_row    = pick_matches.iloc[0] if not pick_matches.empty else df_ranked.iloc[0]
    final_pick   = final_row.to_dict()

    # Handle mock rec: replace placeholder names with real ones
    if rec.get("_mock"):
        rec["top_pick"]  = final_pick["name"]
        rec["runner_up"] = df_ranked.iloc[1]["name"] if len(df_ranked) > 1 else ""

    # ── Save session files ────────────────────────────────────────
    save_to_csv(df_ranked, location, cuisine)
    save_to_json(df_ranked, location, cuisine, user_budget=budget)

    # ── Save to user history if logged in ─────────────────────────
    if current_user.is_authenticated:
        save_to_history(current_user.username, {
            "session_id": session_id,
            "cuisine":    cuisine,
            "location":   location,
            "budget":     budget,
            "top_pick":   final_pick.get("name"),
            "score":      round(float(final_pick.get("score", 0)), 3),
            "saved_at":   datetime.now().isoformat(),
        })

    # ── Cache payload ─────────────────────────────────────────────
    _cache[session_id] = {
        "session_id":     session_id,
        "cuisine":        cuisine,
        "location":       location,
        "budget":         budget,
        "restrictions":   restrictions,
        "stats":          stats,
        "top5":           df_ranked.head(5).to_dict(orient="records"),
        "all_results":    df_ranked.to_dict(orient="records"),
        "final_pick":     final_pick,
        "recommendation": rec,
        "charts":         charts,
        "mode":           mode,
        "searched_at":    datetime.now().strftime("%B %d, %Y at %I:%M %p"),
    }

    return redirect(url_for("results", session_id=session_id))


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/results/<session_id>")
def results(session_id: str):
    payload = _cache.get(session_id)
    if not payload:
        flash("Session expired. Please run a new search.", "warning")
        return redirect(url_for("home"))
    return render_template("results.html", **payload)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user     = login_user_lookup(username, password)
        if user:
            flask_login_user(user, remember=request.form.get("remember") == "on")
            flash(f"Welcome back, {user.username}! 🌺", "success")
            return redirect(request.args.get("next") or url_for("home"))
        flash("Incorrect username or password.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")

        if password != confirm:
            flash("Passwords do not match.", "error")
        else:
            ok, msg = register_user(username, email, password)
            if ok:
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
            flash(msg, "error")

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You've been logged out.", "info")
    return redirect(url_for("home"))


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        location     = request.form.get("location", "").strip()
        budget       = int(request.form.get("budget", 2))
        raw_rest     = request.form.get("restrictions_hidden", "")
        restrictions = [r.strip() for r in raw_rest.split(",") if r.strip()]
        update_profile(current_user.username, location, budget, restrictions)
        flash("Preferences saved!", "success")
        return redirect(url_for("profile"))

    return render_template("profile.html",
                           user=current_user,
                           islands=ISLANDS,
                           restrictions=RESTRICTIONS)


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/history")
@login_required
def history():
    return render_template("history.html",
                           history=list(reversed(current_user.history)))


# ─────────────────────────────────────────────────────────────────────────────
# CHART SERVING
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/chart/<session_id>")
def chart(session_id: str):
    """Serves bar chart PNG inline for <img> tags in results.html."""
    path = os.path.join(os.path.dirname(__file__), "charts", f"bar_{session_id}.png")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/radar/<session_id>")
def radar(session_id: str):
    """Serves the self-contained Plotly HTML loaded into the results iframe."""
    path = os.path.join(os.path.dirname(__file__), "charts", f"radar_{session_id}.html")
    if not os.path.exists(path):
        return ("<p style='text-align:center;padding:3rem;color:#999'>"
                "Radar chart not available</p>"), 404
    return send_file(path, mimetype="text/html")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  🌺  Hawaii Eats — Restaurant Comparison App")
    print("  ─────────────────────────────────────────────")
    print("  Open: http://127.0.0.1:5000")
    print("  Demo mode available — no SerpAPI key needed\n")
    app.run(debug=True, port=5000)
    