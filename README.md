# 🌺 Hawaii Eats — Restaurant Comparison App

> **ITM352 · Spring 2026 · University of Hawaiʻi at Mānoa**  
> Nizhen He · Grace Kulhanek · Sara Bautista

Hawaii Eats is a full-stack Flask web application that helps users find and compare restaurants across the Hawaiian Islands. Enter a cuisine, island, and budget — the app fetches live Yelp data, scores every result using a weighted algorithm, and uses **Claude AI** to pick the single best restaurant for you.

---

## ✨ Features

- 🔍 **Smart Search** — filter by cuisine, island, budget, and dietary restrictions
- 📊 **Weighted Scoring** — composite score: rating (50%) + review volume (30%) + price fit (20%)
- 🤖 **AI Recommendation** — Claude AI reviews the top results and picks one with a written justification
- 📈 **Data Visualizations** — three charts per search: scatter plot, score bar chart, and estimated busy times
- 🤝 **Side-by-Side Compare** — select up to 5 restaurants and compare features in a table
- 👤 **User Accounts** — register, log in, save preferences, and access personalized features
- ⭐ **For You** — personalized picks generated from your search history
- 🕐 **Search History** — log of your last 20 searches with scores
- 💾 **Export** — download results as CSV or JSON
- 🏖️ **Demo Mode** — works fully offline with cached data, no API key required

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask, Flask-Login, Werkzeug |
| Data | pandas, numpy |
| Visualization | matplotlib, plotly |
| AI | Claude API (claude-sonnet) |
| External Data | SerpAPI — Yelp engine |
| Auth Storage | Flat JSON file (no database needed) |
| Frontend | Jinja2 templates, vanilla CSS/JS |

---

## 📁 Project Structure

```
hawaii-eats/
├── requirements.txt          # Python dependencies
└── app/
    ├── app.py                # Flask application & all routes
    ├── pipeline.py           # Data cleaning & DataFrame builder
    ├── scoring.py            # Composite scoring algorithm
    ├── ai_recommender.py     # Claude AI recommendation
    ├── visualizations.py     # Chart generation (matplotlib + plotly)
    ├── scraper.py            # SerpAPI / Yelp data fetcher
    ├── auth.py               # User registration, login, session
    ├── recommendations.py    # For You personalized picks
    ├── file_io.py            # CSV / JSON export & import
    ├── check.py              # Dependency checker utility
    ├── test_backend.py       # Automated backend test suite (24 tests)
    ├── yelp_results.json     # Demo mode dataset
    ├── data/
    │   └── users.json        # User account storage (auto-created)
    ├── sessions/             # Exported CSV/JSON files (auto-created)
    ├── charts/               # Generated chart HTML files (auto-created)
    └── templates/            # Jinja2 HTML templates
        ├── base.html
        ├── home.html
        ├── results.html
        ├── compare.html
        ├── for_you.html
        ├── history.html
        ├── profile.html
        ├── login.html
        ├── register.html
        └── contact.html
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip
- Git

### Steps

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd hawaii-eats
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. (Optional) Verify all packages installed correctly**
```bash
cd app
python check.py
```
You should see a ✅ next to every package name.

**4. Start the app**
```bash
cd app
python app.py
```

**5. Open in your browser**
```
http://127.0.0.1:5000
```

> **To stop the server:** press `Ctrl + C` in the terminal.  
> **macOS / Linux:** use `python3 app.py` if `python` is not recognized.

---

## 🌐 User Guide

### Navigation Bar

| Link | When visible |
|---|---|
| **Search** | Always |
| **For You** ✦ | Logged-in users only |
| **History** | Logged-in users only |
| **Profile** | Logged-in users only |
| **Contact** | Always |
| **Log in / Sign up** | Guests only |
| **Log out** | Logged-in users only |

---

### 🔍 Running a Search

1. Go to **http://127.0.0.1:5000**
2. Choose a **Cuisine** from the dropdown (Japanese, Korean, Hawaiian, Seafood, etc.)
3. Choose an **Island** (Oahu – Honolulu, Waikiki, Kailua · Maui · Big Island · Kauai)
4. Click a **Budget** tile: `$` Under $15 · `$$` $15–30 · `$$$` $30–60 · `$$$$` $60+
5. Optionally check any **Dietary Restrictions** (Vegetarian, Vegan, Gluten-Free, Halal, etc.)
6. Select **Mode:**
   - **Demo** — uses the included `yelp_results.json` dataset, no API key needed ✅
   - **Live** — fetches real-time Yelp data via SerpAPI (requires active API key)
7. Click **🔍 Find My Restaurant**

> **Recommended first test:** Japanese · Honolulu · $$ · Demo mode

---

### 📋 Results Page

| Section | What it shows |
|---|---|
| **AI Recommendation card** | Top pick chosen by Claude AI with a written justification and runner-up |
| **Top 5 cards** | Ranked restaurant cards with photo, rating, price, score bar, and Yelp link |
| **Charts** | Rating vs. Review Count · Score Breakdown · Estimated Busy Times |
| **Full table** | All returned restaurants sorted by score, with Download CSV / JSON buttons |

To compare restaurants, **check up to 5 restaurant cards**, then click **Compare Selected**.

---

### 🤝 Compare Page

Displays selected restaurants side by side:
- **Top row:** photo, score bar, rating, price tier, and neighborhood per restaurant
- **Feature table:** Rating · Reviews · Price · Address · Phone · Website

Click **Back to Results** to return without losing your selections.

---

### 👤 Creating an Account

1. Click **Sign up** in the nav bar
2. Enter a **Username** (3+ characters), **Email**, and **Password** (6+ characters)
3. Confirm your password and click **Create Account**
4. You'll be redirected to the login page — sign in with your new credentials

---

### ⚙️ Profile & Preferences

1. Click **Profile** in the nav bar (login required)
2. Set a default **Island**, **Budget**, and **Dietary Restrictions**
3. Click **Save Preferences** — these will pre-fill the search form on every login

---

### 🕐 Search History

Click **History** in the nav bar to see your last 20 searches. Each entry shows the cuisine, island, budget, top-recommended restaurant, match score, and date.

---

### ⭐ For You — Personalized Picks

Click **For You** in the nav bar (login required). The app analyzes your search history to find your most-searched cuisines, islands, and typical budget, then surfaces 5 personalized restaurant picks. Picks refresh every time you log in. Run at least 1–2 searches first to activate it.

---

### 📬 Contact

Click **Contact** in the nav bar. Fill in your name, email, and message, then click **Send Message** — this opens your email client pre-addressed to the full team.

---

## 🧪 Running the Tests

The automated test suite covers all four backend modules:

```bash
cd app
python test_backend.py
```

Expected output:
```
=======================================================
   Backend Test Suite — Restaurant Comparison App
=======================================================

[ 1 ] pipeline.py
  ✅ DataFrame builds correctly
  ✅ Drops rows missing name/rating
  ✅ Handles empty input gracefully
  ✅ Column types are correct
  ✅ summarize() returns stats

[ 2 ] scoring.py
  ✅ score_restaurants() adds required columns
  ✅ Results sorted best → worst
  ✅ Rank starts at 1
  ✅ All scores between 0 and 1
  ✅ Handles empty DataFrame
  ✅ get_top_n() returns correct count
  ✅ to_recommendation_payload() structure

[ 3 ] file_io.py
  ✅ save_to_csv() creates a file
  ✅ load_from_csv() restores data
  ✅ save_to_json() creates a file
  ✅ load_from_json() restores data
  ✅ load_from_csv() handles missing file
  ✅ list_sessions() returns a list

[ 4 ] scraper.py  (parser & config — no live API call)
  ✅ parse_result() maps all fields correctly
  ✅ parse_result() remaps 'reviews' → 'review_count'
  ✅ parse_result() handles missing fields
  ✅ parse_result() returns all required keys
  ✅ API key is set (not placeholder)
  ✅ parse_result() output works with pipeline

=======================================================
  Results: 24/24 tests passed
  ✅ All tests passed — your backend is ready!
=======================================================
```

> Tests use mock data — no live API calls are made.

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| Browser shows "This site can't be reached" | Server isn't running. Go to `app/` and run `python app.py` again. |
| `ModuleNotFoundError` in terminal | Run `pip install -r requirements.txt` from the repo root. |
| Red banner: "Scrape failed" | Switch to **Demo** mode on the search form. Live mode requires a valid SerpAPI key. |
| Yellow banner: "No restaurants found" | Try a different cuisine + island + budget combo. |
| Yellow banner: "Session expired" | Server was restarted. Run a new search. |
| For You page says "run a few searches first" | Complete 1–2 searches while logged in, then revisit. |
| `python` not recognized (macOS/Linux) | Use `python3 app.py` instead. |
| Port 5000 already in use | Run `flask run --port 5001` from `app/` and visit `http://127.0.0.1:5001`. |

---

## 👥 Team

| Name | Role |
|---|---|
| **Nizhen He** | Backend · Data Pipeline · Visualization · Testing |
| **Sara Bautista** | Frontend · Visualization · AI Integration · Testing |

📧 [Contact us](mailto:nh42@hawaii.edu)

---

*Hawaii Eats — ITM352 Restaurant Comparison App · Spring 2026*
