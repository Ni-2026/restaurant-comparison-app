# visualizations.py
# Sara Bautista — ITM352 Restaurant Comparison App
#
# UPDATED: Replaced the composite-score bar chart with a Peak Business Times
# chart (Plotly HTML) for the AI's top pick. The radar chart is unchanged.

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")


def _ensure_dir():
    os.makedirs(CHARTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PEAK BUSINESS TIMES CHART (replaces the old bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def generate_peak_times_chart(restaurant_name: str,
                               peak_data: dict,
                               save_path: str = None) -> str:
    """
    Renders an interactive Plotly chart showing estimated busy levels by hour
    for a single restaurant — modeled on Google Maps' "Popular Times" widget.

    Args:
        restaurant_name (str):  Name shown in chart title
        peak_data (dict):       Output of scraper.estimate_peak_times()
        save_path (str):        Output path (.html); auto-named if not provided

    Returns:
        str: Path to the saved .html file, or "" if no data
    """
    if not peak_data or "busy_by_hour" not in peak_data:
        print("[Charts] Peak-times chart skipped: no data.")
        return ""

    _ensure_dir()

    busy = peak_data["busy_by_hour"]
    peak_hour = peak_data.get("peak_hour")
    avg_wait  = peak_data.get("avg_wait_min", 0)
    is_est    = peak_data.get("is_estimated", True)

    hours = sorted(busy.keys())
    levels = [busy[h] for h in hours]

    # Hour labels: 11 → "11a", 12 → "12p", 13 → "1p", etc.
    def fmt_hour(h):
        if h == 12:  return "12p"
        if h < 12:   return f"{h}a"
        return f"{h - 12}p"

    labels = [fmt_hour(h) for h in hours]

    # Color: peak hour highlighted in coral, rest in teal
    bar_colors = [
        "#c95032" if h == peak_hour else "#1a7a8f"
        for h in hours
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=levels,
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v}%" for v in levels],
        textposition="outside",
        textfont=dict(size=9, color="#666"),
        hovertemplate="<b>%{x}</b><br>%{y}% busy<extra></extra>",
    ))

    # Title with the wait-time + transparency note
    est_note = "Estimated from typical patterns" if is_est else ""
    title_html = (
        f"<b>{restaurant_name}</b><br>"
        f"<span style='font-size:11px;color:#666;'>"
        f"Peak: {peak_data.get('peak_label','')} "
        f"&middot; ~{avg_wait} min wait at peak"
        f"{' &middot; ' + est_note if est_note else ''}"
        f"</span>"
    )

    fig.update_layout(
        title=dict(text=title_html, font=dict(size=14, family="Inter")),
        xaxis=dict(title="", tickfont=dict(size=10), showgrid=False),
        yaxis=dict(
            title="Busy %",
            range=[0, 115],
            tickfont=dict(size=9),
            gridcolor="#eee",
            zeroline=False,
        ),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=80, b=40, l=50, r=20),
        height=420,
        bargap=0.25,
    )

    if not save_path:
        save_path = os.path.join(CHARTS_DIR, "peak_times.html")
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"[Charts] Peak-times chart → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def generate_radar_chart(df: pd.DataFrame, restaurant_names: list = None,
                          save_path: str = None) -> str:
    """
    Interactive Plotly radar chart comparing up to 4 restaurants across 5 axes.
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

    max_rev = df["review_count"].max() or 1
    subset["_rev_display"] = subset["review_count"] / max_rev

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
                         top_pick_name: str = "",
                         peak_data: dict = None,
                         cuisine: str = "",
                         location: str = "") -> dict:
    """
    Generates both charts with session-scoped filenames.

    Args:
        df              (DataFrame):  Scored & ranked restaurants
        session_id      (str):        Used to namespace output filenames
        top_pick_name   (str):        Restaurant name for the peak-times chart
        peak_data       (dict):       Output of scraper.estimate_peak_times()
        cuisine, location (str):      Used for the chart title

    Returns:
        dict: {"peak": path_to_html, "radar": path_to_html}
    """
    peak_path = ""
    if peak_data and top_pick_name:
        peak_path = generate_peak_times_chart(
            top_pick_name,
            peak_data,
            save_path=os.path.join(CHARTS_DIR, f"peak_{session_id}.html"),
        )

    radar_path = generate_radar_chart(
        df,
        save_path=os.path.join(CHARTS_DIR, f"radar_{session_id}.html"),
    )
    return {"peak": peak_path, "radar": radar_path}