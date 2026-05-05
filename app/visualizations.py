# visualizations.py
# Sara Bautista — ITM352 Restaurant Comparison App
#
# Generates two chart types from a scored restaurant DataFrame:
#   generate_bar_chart()   — horizontal bar chart (top-N scores) → .png
#   generate_radar_chart() — multi-axis radar chart (head-to-head) → .html

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for Flask (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")


def _ensure_dir():
    os.makedirs(CHARTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def generate_bar_chart(df: pd.DataFrame, top_n: int = 5,
                        title: str = "Top Restaurants by Score",
                        save_path: str = None) -> str:
    """
    Horizontal bar chart of composite scores for the top N restaurants.
    Bars are color-coded by price tier so budget is visible at a glance.

    Args:
        df        (pd.DataFrame): Scored and ranked restaurant DataFrame
        top_n     (int):          Number of restaurants to show
        title     (str):          Chart title
        save_path (str):          If given, saves to this path; otherwise auto-names

    Returns:
        str: Absolute path to the saved .png, or "" on failure
    """
    if df.empty:
        print("[Charts] Bar chart skipped: empty DataFrame.")
        return ""

    _ensure_dir()
    data = df.head(top_n).copy()

    # Color map: each price tier gets a distinct color
    price_colors = {"$": "#4CAF50", "$$": "#2196F3", "$$$": "#FF9800",
                    "$$$$": "#F44336", "N/A": "#9E9E9E"}
    colors = [price_colors.get(str(p), "#9E9E9E") for p in data["price"]]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.85)))

    # Reverse so rank 1 appears at the top of the chart
    bars = ax.barh(data["name"][::-1], data["score"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.8, height=0.6)

    # Score label to the right of each bar
    for bar, score in zip(bars, data["score"][::-1]):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", ha="left", fontsize=9, color="#444")

    ax.set_xlim(0, 1.14)
    ax.set_xlabel("Composite Score  (rating 50% · reviews 30% · price fit 20%)",
                  fontsize=9, color="#555")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Price legend — escape $ for matplotlib's math parser
    label_map = {"$": r"\$", "$$": r"\$\$", "$$$": r"\$\$\$", "$$$$": r"\$\$\$\$"}
    patches = [mpatches.Patch(color=c, label=label_map.get(p, p))
               for p, c in price_colors.items() if p != "N/A"]
    ax.legend(handles=patches, title="Price Tier", loc="lower right",
              fontsize=8, title_fontsize=8)

    plt.tight_layout()

    if not save_path:
        save_path = os.path.join(CHARTS_DIR, "bar_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Charts] Bar chart → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def generate_radar_chart(df: pd.DataFrame, restaurant_names: list = None,
                          save_path: str = None) -> str:
    """
    Interactive Plotly radar chart comparing up to 4 restaurants across 5 axes:
      Rating · Review Volume · Price Fit · Composite Score · Value

    Saved as a self-contained HTML file with Plotly CDN — opens in any browser.

    Args:
        df               (pd.DataFrame): Scored DataFrame
        restaurant_names (list[str]):    Names to compare; defaults to top 4
        save_path        (str):          Output path; auto-named if not given

    Returns:
        str: Absolute path to the saved .html, or "" on failure
    """
    if df.empty:
        print("[Charts] Radar chart skipped: empty DataFrame.")
        return ""

    _ensure_dir()

    names = restaurant_names or df["name"].head(4).tolist()
    subset = df[df["name"].isin(names)].copy()
    if subset.empty:
        return ""

    axes = ["Rating", "Review Volume", "Price Fit", "Composite Score", "Value"]

    # Normalize review count to 0-1 for the radar display
    max_rev = df["review_count"].max() or 1
    subset["_rev_display"] = subset["review_count"] / max_rev

    # Value = review volume / price tier  (high reviews, low cost = best value)
    subset["_value"] = subset.apply(
        lambda r: r["_rev_display"] / max(r["price_num"], 1), axis=1)
    v_max = subset["_value"].max() or 1
    subset["_value"] = subset["_value"] / v_max

    COLORS = ["#0a6e6e", "#d95a3a", "#2196F3", "#9C27B0"]
    fig = go.Figure()

    for i, (_, row) in enumerate(subset.iterrows()):
        vals = [
            float(row["rating_norm"]),
            float(row["_rev_display"]),
            float(row["price_fit"]),
            float(row["score"]),
            float(row["_value"]),
        ]
        # Close the polygon by repeating the first value
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=axes + [axes[0]],
            fill="toself",
            name=row["name"],
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            fillcolor=COLORS[i % len(COLORS)],
            opacity=0.2,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        title=dict(text="Head-to-Head Comparison", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28),
        margin=dict(t=70, b=90),
        width=600, height=500,
        paper_bgcolor="white",
    )

    if not save_path:
        save_path = os.path.join(CHARTS_DIR, "radar_chart.html")
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"[Charts] Radar chart → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE — generate both at once
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_charts(df: pd.DataFrame, session_id: str,
                         top_n: int = 5, cuisine: str = "",
                         location: str = "") -> dict:
    """
    Generates both charts with session-scoped filenames so concurrent searches
    don't overwrite each other.

    Returns:
        dict: {"bar": path_to_png, "radar": path_to_html}
    """
    label    = f"Top {top_n} {cuisine.title()} Restaurants — {location}" if cuisine else "Top Restaurants"
    bar_path = generate_bar_chart(
        df, top_n=top_n, title=label,
        save_path=os.path.join(CHARTS_DIR, f"bar_{session_id}.png"),
    )
    radar_path = generate_radar_chart(
        df,
        save_path=os.path.join(CHARTS_DIR, f"radar_{session_id}.html"),
    )
    return {"bar": bar_path, "radar": radar_path}
