"""Plotly figures shared by the app pages.

Figures do not hard-code background or text colours: they are rendered with
Streamlit's theme (``st.plotly_chart(..., theme="streamlit")``), so they follow
the light/dark setting. Only semantic colours (status, actual vs forecast) are
fixed, and they were chosen to read on both backgrounds.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src import config as C

# Semantic colours readable on both the light (#FFFFFF) and dark (#0B1424) background.
STATUS_HEX = {"green": "#12A87A", "blue": "#2F6FE4", "orange": "#E39B1A", "gray": "#8494AD",
              "violet": "#8B7CF6", "red": "#E5484D"}
OCC_HEX = {"LOW": "#12A87A", "MODERATE": "#E3B21A", "HIGH": "#E07B1A", "FULL": "#E5484D"}
ACTUAL = "#8494AD"


def _base(fig: go.Figure, height=360, title=None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=44 if title else 12, b=8),
                      title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
                      hoverlabel=dict(font_size=13))
    return fig


def map_style(theme_type: str | None, offline: bool = False):
    """CARTO street tiles (loaded by the browser, need internet) or a plain
    background that works offline. Plain mode keeps the markers, legend and
    hover details, but shows no streets."""
    dark = theme_type == "dark"
    if offline:
        return {"version": 8, "sources": {}, "layers": [
            {"id": "background", "type": "background", "paint": {"background-color": "#101B2E" if dark else "#EEF2F7"}}]}
    return "carto-darkmatter" if dark else "carto-positron"


def facility_map(fac: pd.DataFrame, theme_type: str | None, height=460, highlight: str | None = None,
                 destination: dict | None = None, radius_km: float | None = None, offline: bool = False) -> go.Figure:
    """Map of facilities with verified coordinates, coloured by evidence status."""
    style = map_style(theme_type, offline)
    pts = fac[fac["has_location"]]
    fig = go.Figure()
    for label, grp in pts.groupby("status_label", sort=False):
        color = STATUS_HEX[grp["status_color"].iloc[0]]
        sizes = [17 if r == highlight else 12 for r in grp["id"]]
        fig.add_trace(go.Scattermap(
            lat=grp["lat"], lon=grp["lon"], mode="markers", name=label,
            marker=dict(size=sizes, color=color, opacity=0.95),
            customdata=grp[["name", "type_label", "locality", "coord_precision"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>"
                          "<i>%{customdata[3]}</i><extra>" + label + "</extra>"))
    center = dict(lat=18.545, lon=73.84)
    zoom = 10.6
    if destination and destination.get("lat") is not None and not pd.isna(destination.get("lat")):
        fig.add_trace(go.Scattermap(lat=[destination["lat"]], lon=[destination["lon"]], mode="markers",
                                    name="Your destination",
                                    marker=dict(size=15, color="#5B4FD6", symbol="circle"),
                                    hovertemplate=f"<b>{destination['name']}</b><extra>Destination</extra>"))
        center = dict(lat=destination["lat"], lon=destination["lon"])
        zoom = 12.2 if (radius_km or 5) <= 3 else 11.4
    fig.update_layout(map=dict(style=style, center=center, zoom=zoom), height=height,
                      margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01, bgcolor="rgba(127,127,127,0.12)"))
    return fig


def daily_trend(grid: pd.DataFrame) -> go.Figure:
    obs = grid.dropna(subset=["occupancy_rate"])
    daily = obs.groupby("date")["occupancy_rate"].mean() * 100
    fig = go.Figure(go.Scatter(x=daily.index, y=daily.values, mode="lines+markers", name="Daily mean",
                               line=dict(width=2.4), marker=dict(size=4),
                               hovertemplate="%{x|%a %d %b}: %{y:.1f}%<extra></extra>"))
    for d, lab in ((C.TRAIN_END, "training ends"), (C.VAL_END, "test starts")):
        fig.add_vline(x=pd.Timestamp(d) + pd.Timedelta(hours=12), line_dash="dot", line_color=ACTUAL)
        fig.add_annotation(x=pd.Timestamp(d), y=1.02, yref="paper", text=lab, showarrow=False, font=dict(size=11))
    fig.update_yaxes(title="Mean occupancy (%)")
    return _base(fig, 300)


def hour_profile(grid: pd.DataFrame) -> go.Figure:
    obs = grid.dropna(subset=["occupancy_rate"]).assign(hour=lambda d: 8 + d["slot"] * 0.5,
                                                         weekend=lambda d: d["date"].dt.dayofweek >= 5)
    fig = go.Figure()
    for flag, name in ((False, "Weekdays"), (True, "Weekends")):
        s = obs[obs["weekend"] == flag].groupby("hour")["occupancy_rate"].mean() * 100
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name,
                                 customdata=[fmt_time(h) for h in s.index],
                                 line=dict(width=2.6, dash="solid" if not flag else "dot"),
                                 hovertemplate="%{customdata}: %{y:.1f}%<extra>" + name + "</extra>"))
    fig.update_yaxes(title="Mean occupancy (%)", rangemode="tozero")
    fig.update_xaxes(tickvals=list(range(8, 17, 2)), ticktext=[f"{h:02d}:00" for h in range(8, 17, 2)])
    return _base(fig, 300)


def forecast_day(actual: pd.DataFrame, tables: dict, park: str, date) -> go.Figure:
    """Actual occupancy for one car park/day with 30- and 60-minute forecasts and the 60-min range."""
    fig = go.Figure()
    t60 = tables.get(60)
    if t60 is not None:
        d = t60[(t60["SystemCodeNumber"] == park) & (t60["date"].dt.date == date)].sort_values("target_time")
        fig.add_trace(go.Scatter(x=list(d["target_time"]) + list(d["target_time"])[::-1],
                                 y=list(d["pred_high"] * 100) + list(d["pred_low"] * 100)[::-1], fill="toself",
                                 fillcolor="rgba(47,111,228,0.16)", line=dict(width=0), name="60-min 90% range",
                                 hoverinfo="skip"))
    for h, dash in ((60, "solid"), (30, "dot")):
        t = tables.get(h)
        if t is None:
            continue
        d = t[(t["SystemCodeNumber"] == park) & (t["date"].dt.date == date)].sort_values("target_time")
        fig.add_trace(go.Scatter(x=d["target_time"], y=d["pred_rate"] * 100, mode="lines+markers",
                                 name=f"Forecast made {h} min earlier", line=dict(width=2.2, dash=dash),
                                 marker=dict(size=5), hovertemplate="%{x|%H:%M}: %{y:.1f}%<extra>" + f"{h}-min forecast</extra>"))
    fig.add_trace(go.Scatter(x=actual["slot_time"], y=actual["occupancy_rate"] * 100, mode="lines+markers", name="Actual",
                             line=dict(width=3, color=ACTUAL), marker=dict(size=6),
                             hovertemplate="%{x|%H:%M}: %{y:.1f}%<extra>Actual</extra>"))
    for lab, b in zip(C.STATUS_LABELS[1:], C.STATUS_BINS):
        fig.add_hline(y=b * 100, line_dash="dash", line_color=OCC_HEX[lab], opacity=0.55,
                      annotation_text=f"{lab} ≥ {int(b * 100)}%", annotation_position="top left",
                      annotation_font_color=OCC_HEX[lab], annotation_font_size=11)
    fig.update_yaxes(title="Occupancy (%)", range=[0, 105])
    return _base(fig, 420)


def weekly_heatmap(hist: pd.DataFrame) -> go.Figure:
    hm = hist.assign(dow=hist["date"].dt.dayofweek, hour=8 + hist["slot"] * 0.5).groupby(
        ["dow", "hour"])["occupancy_rate"].mean().unstack() * 100
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Heatmap(z=hm.values, x=[fmt_time(h) for h in hm.columns], y=[days[i] for i in hm.index],
                               zmin=0, zmax=100, colorbar=dict(title="%", thickness=12),
                               hovertemplate="%{y} %{x}: %{z:.0f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 290)


def fmt_time(h: float) -> str:
    return f"{int(h):02d}:{int(round((h % 1) * 60)):02d}"


# ----------------------------------------------------------------------------
# Model-performance charts (built from outputs/metrics and outputs/predictions)
# ----------------------------------------------------------------------------
def model_comparison(reg: pd.DataFrame) -> go.Figure:
    d = reg.sort_values("Test MAE (pp)", ascending=True)
    colors = ["#2F6FE4" if s else "#8494AD" for s in d["Selected"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d["Model"], x=d["Test MAE (pp)"], orientation="h", marker_color=colors, name="Test MAE",
                         hovertemplate="%{y}<br>Test MAE %{x:.2f} pp<extra></extra>"))
    fig.add_trace(go.Scatter(y=d["Model"], x=d["Walk-forward MAE (pp)"], mode="markers", name="Walk-forward mean ± sd",
                             marker=dict(size=10, color="#E39B1A", symbol="diamond"),
                             error_x=dict(type="data", array=d["Walk-forward MAE std (pp)"], thickness=1.5),
                             hovertemplate="%{y}<br>Walk-forward %{x:.2f} pp<extra></extra>"))
    fig.update_xaxes(title="Mean absolute error (percentage points, lower is better)")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 330)


def confusion(cm: list, labels: list, title: str) -> go.Figure:
    import numpy as np
    m = np.array(cm, dtype=float)
    norm = m / m.sum(axis=1, keepdims=True)
    text = [[f"{int(m[i, j])}<br>{norm[i, j]:.0%}" for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels, text=text, texttemplate="%{text}", zmin=0, zmax=1,
                               showscale=False, hovertemplate="Actual %{y}, predicted %{x}<br>%{text}<extra></extra>"))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return _base(fig, 330, title)


def actual_vs_predicted(tp: pd.DataFrame, col: str) -> go.Figure:
    s = tp.sample(min(len(tp), 3000), random_state=0)
    fig = go.Figure(go.Scattergl(x=s["target_rate"] * 100, y=s[col] * 100, mode="markers",
                                 marker=dict(size=4, opacity=0.35, color="#2F6FE4"), name="Test forecasts",
                                 hovertemplate="Actual %{x:.1f}%<br>Predicted %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color=ACTUAL),
                             name="Perfect forecast", hoverinfo="skip"))
    fig.update_xaxes(title="Actual occupancy (%)", range=[0, 102])
    fig.update_yaxes(title="Predicted occupancy (%)", range=[0, 102])
    return _base(fig, 360)


def importance(imp: pd.DataFrame) -> go.Figure:
    d = imp.sort_values("importance_MAE_increase_pp")
    fig = go.Figure(go.Bar(y=d["feature"], x=d["importance_MAE_increase_pp"], orientation="h",
                           error_x=dict(type="data", array=d["std_pp"]), marker_color="#2F6FE4",
                           hovertemplate="%{y}: +%{x:.2f} pp when shuffled<extra></extra>"))
    fig.update_xaxes(title="Increase in test MAE when the feature is shuffled (pp)", type="log")
    return _base(fig, 420)


def error_by_hour(tp: pd.DataFrame, sel_col: str) -> go.Figure:
    d = tp.assign(hour=8 + tp["target_slot"] * 0.5)
    fig = go.Figure()
    for col, name, dash in ((sel_col, "Selected model", "solid"),
                            ("reg::Persistence + weekly shape (baseline)", "Persistence + weekly shape", "dot"),
                            ("reg::Persistence (baseline)", "Persistence", "dash")):
        e = (d["target_rate"] - d[col]).abs().groupby(d["hour"]).mean() * 100
        fig.add_trace(go.Scatter(x=e.index, y=e.values, mode="lines+markers", name=name,
                                 customdata=[fmt_time(h) for h in e.index],
                                 line=dict(dash=dash, width=2.4 if dash == "solid" else 1.6),
                                 hovertemplate="%{customdata}: %{y:.2f} pp<extra>" + name + "</extra>"))
    fig.update_xaxes(tickvals=list(range(9, 17)), ticktext=[f"{h:02d}:00" for h in range(9, 17)])
    fig.update_yaxes(title="MAE (pp)", rangemode="tozero")
    fig.update_xaxes(title="Time being forecast")
    return _base(fig, 330)


def per_park(pp: pd.DataFrame) -> go.Figure:
    d = pp.sort_values("MAE_pp")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d["SystemCodeNumber"], x=d["persistence_MAE_pp"], orientation="h", name="Persistence",
                         marker_color="#8494AD", opacity=0.55,
                         hovertemplate="%{y}: %{x:.2f} pp<extra>Persistence</extra>"))
    fig.add_trace(go.Bar(y=d["SystemCodeNumber"], x=d["MAE_pp"], orientation="h", name="Selected model",
                         marker_color="#2F6FE4", hovertemplate="%{y}: %{x:.2f} pp<extra>Selected model</extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Test MAE (pp)")
    return _base(fig, 620)


def interval_coverage(report: dict) -> go.Figure:
    bands = list(report["by_band"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bands, y=[report["by_band"][b]["global_coverage"] * 100 for b in bands],
                         name="One width for all times", marker_color="#8494AD",
                         hovertemplate="%{x}: %{y:.1f}%<extra>Global width</extra>"))
    fig.add_trace(go.Bar(x=bands, y=[report["by_band"][b]["band_coverage"] * 100 for b in bands],
                         name="Width by time of day", marker_color="#2F6FE4",
                         hovertemplate="%{x}: %{y:.1f}%<extra>Band width</extra>"))
    fig.add_hline(y=90, line_dash="dash", line_color="#E39B1A", annotation_text="target 90%",
                  annotation_position="top left")
    fig.update_yaxes(title="Test outcomes inside the range (%)", range=[80, 100])
    fig.update_xaxes(title="Time being forecast")
    return _base(fig, 320)
