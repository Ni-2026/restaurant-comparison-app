"""
app.py
------
Main Flask application. Handles all routes, ties together the pipeline,
and renders HTML templates.

Routes:
  GET  /              → search form (index)
  POST /results       → run pipeline, show results
  GET  /sessions      → list saved sessions
  POST /load          → load a saved session and show results
  GET  /download/<f>  → download a session file
"""

import os
import logging
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash,
    session,
)
from dotenv import load_dotenv

from scraper_api import scrape_yelp          # swap to scraper_bs if needed
from data_pipeline import run_pipeline
from visualizations import generate_bar_chart, generate_radar_chart
from ai_recommender import get_recommendation
from file_io import save_session, load_session, list_sessions

load_dotenv()

# ── App Setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SESSION_DIR = "sessions"


# ── Helper ─────────────────────────────────────────────────────────────────────

def _df_to_records(df):
    """Converts DataFrame to a list of dicts for Jinja templates."""
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Home page — search form."""
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results():
    """
    Handles form submission:
      1. Validates inputs
      2. Scrapes Yelp
      3. Runs data pipeline
      4. Generates charts + AI recommendation
      5. Saves session
      6. Renders results page
    """
    location    = request.form.get("location", "").strip()
    cuisine     = request.form.get("cuisine", "").strip()
    budget      = request.form.get("budget", "").strip()
    max_results = request.form.get("max_results", "10").strip()

    # ── Input validation ──
    errors = []
    if not location:
        errors.append("Please enter a location.")
    if not cuisine:
        errors.append("Please enter a cuisine type.")
    try:
        max_results = int(max_results)
        if max_results < 1 or max_results > 50:
            raise ValueError
    except ValueError:
        errors.append("Number of results must be between 1 and 50.")
        max_results = 10

    if errors:
        for e in errors:
            flash(e, "error")
        return redirect(url_for("index"))

    # ── Scrape ──
    logger.info("Search: '%s' in '%s' (budget: %s, max: %d)", cuisine, location, budget, max_results)
    try:
        restaurants = scrape_yelp(
            location=location,
            cuisine=cuisine,
            budget=budget,
            max_results=max_results,
        )
    except Exception as exc:
        logger.error("Scrape failed: %s", exc)
        flash(f"Could not fetch restaurant data: {exc}", "error")
        return redirect(url_for("index"))

    if not restaurants:
        flash("No restaurants found. Try a different location or cuisine.", "warning")
        return redirect(url_for("index"))

    # ── Pipeline ──
    df = run_pipeline(restaurants, budget=budget, max_results=max_results)

    if df.empty:
        flash("No results after processing. Please try again.", "warning")
        return redirect(url_for("index"))

    # ── Charts ──
    bar_chart   = generate_bar_chart(df, top_n=min(10, len(df)))
    radar_chart = generate_radar_chart(df, top_n=min(5, len(df)))

    # ── AI Recommendation ──
    recommendation = get_recommendation(df, location, cuisine, budget)

    # ── Save Session ──
    saved_paths = {}
    try:
        saved_paths = save_session(
            df,
            location=location,
            cuisine=cuisine,
            budget=budget,
            session_dir=SESSION_DIR,
        )
    except Exception as exc:
        logger.warning("Could not save session: %s", exc)

    return render_template(
        "results.html",
        restaurants=_df_to_records(df),
        bar_chart=bar_chart,
        radar_chart=radar_chart,
        recommendation=recommendation,
        location=location,
        cuisine=cuisine,
        budget=budget,
        total=len(df),
        saved_csv=saved_paths.get("csv", ""),
        saved_json=saved_paths.get("json", ""),
    )


@app.route("/sessions", methods=["GET"])
def sessions_page():
    """Lists all saved sessions."""
    all_sessions = list_sessions(SESSION_DIR)
    return render_template("sessions.html", sessions=all_sessions)


@app.route("/load", methods=["POST"])
def load():
    """Loads a saved session file and re-renders the results page."""
    file_path = request.form.get("file_path", "").strip()

    if not file_path or not Path(file_path).exists():
        flash("Session file not found.", "error")
        return redirect(url_for("sessions_page"))

    try:
        df, metadata = load_session(file_path)
    except Exception as exc:
        flash(f"Could not load session: {exc}", "error")
        return redirect(url_for("sessions_page"))

    if df.empty:
        flash("Session file is empty.", "warning")
        return redirect(url_for("sessions_page"))

    bar_chart   = generate_bar_chart(df, top_n=min(10, len(df)))
    radar_chart = generate_radar_chart(df, top_n=min(5, len(df)))

    return render_template(
        "results.html",
        restaurants=_df_to_records(df),
        bar_chart=bar_chart,
        radar_chart=radar_chart,
        recommendation=None,   # no AI regen on load
        location=metadata.get("location", ""),
        cuisine=metadata.get("cuisine", ""),
        budget=metadata.get("budget", ""),
        total=len(df),
        saved_csv=file_path if file_path.endswith(".csv") else "",
        saved_json=file_path if file_path.endswith(".json") else "",
        loaded=True,
    )


@app.route("/download/<path:filename>", methods=["GET"])
def download(filename):
    """Serves a session file for download."""
    file_path = Path(filename)
    if not file_path.exists():
        flash("File not found.", "error")
        return redirect(url_for("sessions_page"))
    return send_file(str(file_path.resolve()), as_attachment=True)


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path(SESSION_DIR).mkdir(exist_ok=True)
    app.run(debug=os.environ.get("FLASK_ENV") != "production")
