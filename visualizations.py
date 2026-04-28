"""
visualizations.py
-----------------
Generates bar and radar charts from ranked restaurant data.
Uses Plotly so charts are interactive in the browser and return
as HTML strings that Flask can inject directly into templates.

Main functions:
  - generate_bar_chart()   → top N scores as a horizontal bar chart
  - generate_radar_chart() → head-to-head comparison of top restaurants
"""

import logging
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

# ── Design Tokens ──────────────────────────────────────────────────────────────

COLORS = {
    "gold":       "#F4A136",
    "silver":     "#9BA8B5",
    "bronze":     "#C4845A",
    "default":    "#4E8098",
    "background": "#1A1A2E",
    "surface":    "#16213E",
    "text":       "#E8E8E8",
    "grid":       "#2A2A4A",
}

BAR_COLORS = [COLORS["gold"], COLORS["silver"], COLORS["bronze"]]

CHART_CONFIG = {
    "displayModeBar": False,   # hide the plotly toolbar
    "responsive":     True,
}


def _bar_color(rank: int) -> str:
    """Returns gold/silver/bronze for ranks 1-3, default teal for the rest."""
    return BAR_COLORS[rank - 1] if rank <= 3 else COLORS["default"]


# ── Bar Chart ──────────────────────────────────────────────────────────────────

def generate_bar_chart(
    df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Top Restaurants by Score",
) -> Optional[str]:
    """
    Generates a horizontal bar chart of the top N restaurants by score.

    Args:
        df:    Ranked DataFrame with columns: name, score, rank.
        top_n: Number of restaurants to display.
        title: Chart title string.

    Returns:
        HTML string of the interactive Plotly chart.
        Returns None if df is empty or missing required columns.
    """
    required = {"name", "score"}
    if df is None or df.empty or not required.issubset(df.columns):
        logger.warning("generate_bar_chart: missing data or columns.")
        return None

    data = df.head(top_n).copy()
    ranks = data.get("rank", range(1, len(data) + 1))
    bar_colors = [_bar_color(int(r)) for r in ranks]

    fig = go.Figure(
        go.Bar(
            x=data["score"],
            y=data["name"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            text=[f"{s:.1f}" for s in data["score"]],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=12),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Score: %{x:.1f}/100<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=COLORS["text"], size=18),
            x=0.01,
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(
            range=[0, 110],
            gridcolor=COLORS["grid"],
            title="Score (out of 100)",
            title_font=dict(color=COLORS["text"]),
            tickfont=dict(color=COLORS["text"]),
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text"], size=12),
        ),
        margin=dict(l=20, r=40, t=60, b=40),
        height=max(300, top_n * 48),
    )

    return fig.to_html(full_html=False, config=CHART_CONFIG)


# ── Radar Chart ────────────────────────────────────────────────────────────────

def generate_radar_chart(
    df: pd.DataFrame,
    top_n: int = 5,
    title: str = "Head-to-Head Comparison",
) -> Optional[str]:
    """
    Generates a radar chart comparing the top N restaurants across key metrics.

    Metrics compared: Rating, Popularity (reviews), Price Value, Score.

    Args:
        df:    Ranked DataFrame with columns: name, rating, reviews,
               price_level, score.
        top_n: Number of restaurants to include in the comparison.
        title: Chart title string.

    Returns:
        HTML string of the interactive Plotly chart.
        Returns None if df is empty or missing required columns.
    """
    required = {"name", "rating", "reviews", "score"}
    if df is None or df.empty or not required.issubset(df.columns):
        logger.warning("generate_radar_chart: missing data or columns.")
        return None

    data = df.head(top_n).copy()

    # Normalize each metric to 0-100 for fair comparison on the radar
    def norm(series):
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series([50.0] * len(series), index=series.index)
        return ((series - lo) / (hi - lo) * 100).round(1)

    data["rating_norm"]  = norm(data["rating"])
    data["reviews_norm"] = norm(data["reviews"].apply(lambda x: x ** 0.5))  # sqrt scale
    data["score_norm"]   = norm(data["score"])

    # Price value: lower price = better value score (inverted)
    if "price_level" in data.columns:
        pl = data["price_level"].replace(0, 2)   # treat unknown as $$
        data["value_norm"] = norm(5 - pl)
    else:
        data["value_norm"] = pd.Series([50.0] * len(data), index=data.index)

    categories = ["Rating", "Popularity", "Value", "Overall Score"]
    radar_colors = px.colors.qualitative.Bold[:top_n]

    fig = go.Figure()

    for i, (_, row) in enumerate(data.iterrows()):
        values = [
            row["rating_norm"],
            row["reviews_norm"],
            row["value_norm"],
            row["score_norm"],
        ]
        values_closed = values + [values[0]]   # close the polygon
        cats_closed   = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor=radar_colors[i % len(radar_colors)].replace("rgb", "rgba").replace(")", ", 0.15)"),
                line=dict(color=radar_colors[i % len(radar_colors)], width=2),
                name=row["name"],
                hovertemplate=(
                    f"<b>{row['name']}</b><br>"
                    "Rating: %{r:.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=COLORS["text"], size=18),
            x=0.01,
        ),
        polar=dict(
            bgcolor=COLORS["surface"],
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=COLORS["grid"],
                tickfont=dict(color=COLORS["text"], size=10),
                tickvals=[25, 50, 75, 100],
            ),
            angularaxis=dict(
                tickfont=dict(color=COLORS["text"], size=13),
                gridcolor=COLORS["grid"],
            ),
        ),
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        legend=dict(
            font=dict(color=COLORS["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=60, t=60, b=40),
        height=480,
    )

    return fig.to_html(full_html=False, config=CHART_CONFIG)
