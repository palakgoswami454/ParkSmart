"""ParkSmart - Streamlit application.

Run:  streamlit run app.py

All numbers on every page are read from files produced by the pipeline
(data/processed, models, outputs/metrics). Nothing is hard-coded, and nothing
is a live sensor reading: the app replays the 2016 Birmingham data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C

st.set_page_config(page_title="ParkSmart", page_icon="🅿️", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Visual identity: UK car-park signage. Sign blue, road-marking yellow,
# asphalt grey, signal colours for status. Overpass (a Highway-Gothic style
# face) for headings and numbers, Atkinson Hyperlegible for reading text.
# ----------------------------------------------------------------------------
BLUE, ASPHALT, YELLOW, PAPER, MUTED, LINE = "#0B4F9C", "#22272B", "#F2C230", "#F6F7F5", "#5F6873", "#D9DEE3"
STATUS = {"LOW": "#2E9D57", "MODERATE": "#D99A0B", "HIGH": "#E06A1B", "FULL": "#C62828"}
STATUS_TEXT = {"LOW": "Plenty of space", "MODERATE": "Filling up", "HIGH": "Busy", "FULL": "Full or nearly full"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Overpass:wght@400;600;800;900&family=Atkinson+Hyperlegible:wght@400;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, .stCaption {{ font-family: 'Atkinson Hyperlegible', sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Overpass', sans-serif !important; color: {ASPHALT}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 900 !important; }}
[data-testid="stSidebar"] {{ background: {BLUE}; }}
[data-testid="stSidebar"] * {{ color: #ffffff !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(255,255,255,0.16); border-left: 4px solid {YELLOW}; }}
.ps-brand {{ display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.2rem 0; }}
[data-testid="stSidebar"] .ps-p {{ color:{BLUE} !important; }}
.ps-p {{ width:44px; height:44px; border-radius:6px; background:#fff; color:{BLUE}; font:900 32px/44px 'Overpass',sans-serif; text-align:center; }}
.ps-name {{ font:800 22px 'Overpass',sans-serif; color:#fff; line-height:1.05; }}
.ps-sub {{ font-size:12px; opacity:.85; }}
.ps-note {{ border-left:5px solid {YELLOW}; background:#FFF8DD; padding:.65rem .9rem; border-radius:4px; color:{ASPHALT}; font-size:.92rem; margin:.4rem 0 1rem 0; }}
.ps-vms {{ background:{ASPHALT}; border-radius:10px; padding:1.3rem 1.6rem; color:#fff; border:3px solid #3A4147; }}
.ps-vms-top {{ display:flex; justify-content:space-between; font:600 14px 'Overpass',sans-serif; color:#C9D1D8; }}
.ps-vms-num {{ font:900 92px/1 'Overpass',sans-serif; color:{YELLOW}; font-variant-numeric: tabular-nums; margin:.5rem 0 .1rem 0; text-shadow:0 0 18px rgba(242,194,48,.25); }}
.ps-vms-unit {{ font:800 22px 'Overpass',sans-serif; color:{YELLOW}; }}
.ps-vms-status {{ display:inline-block; margin-top:.9rem; padding:.3rem .8rem; border-radius:4px; font:800 16px 'Overpass',sans-serif; color:#fff; }}
.ps-vms-foot {{ margin-top:.8rem; font-size:13px; color:#AEB8C1; }}
.ps-kv {{ font-size:.95rem; color:{ASPHALT}; }}
.ps-kv b {{ font-family:'Overpass',sans-serif; }}
[data-testid="stMetricValue"] {{ font-family:'Overpass',sans-serif; font-weight:800; color:{ASPHALT}; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(font=dict(family="Atkinson Hyperlegible, sans-serif", color=ASPHALT, size=13),
                   plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10),
                   xaxis=dict(gridcolor=LINE, zeroline=False), yaxis=dict(gridcolor=LINE, zeroline=False),
                   legend=dict(orientation="h", y=-0.2), hoverlabel=dict(font_family="Atkinson Hyperlegible"))


def fig_style(fig, title=None, height=360):
    fig.update_layout(**PLOT_LAYOUT, height=height, title=dict(text=title, font=dict(family="Overpass, Atkinson Hyperlegible, sans-serif", size=16)) if title else None)
    return fig


# ----------------------------------------------------------------------------
# Data access (cached). Every loader fails loudly with a fix instruction.
# ----------------------------------------------------------------------------
class MissingArtifact(Exception):
    pass


def _need(path: Path, how: str):
    if not path.exists():
        raise MissingArtifact(f"`{path.relative_to(C.ROOT)}` is missing. {how}")


@st.cache_data(show_spinner="Loading the 30-minute occupancy grid ...")
def get_grid():
    _need(C.GRID_CSV, "Run `python -m src.data_preprocessing`.")
    from src.data_preprocessing import load_grid
    return load_grid()


@st.cache_resource(show_spinner="Loading trained model ...")
def get_bundle(h: int):
    from src.predict import ModelNotFoundError, load_bundle
    try:
        return load_bundle(h)
    except ModelNotFoundError as e:
        raise MissingArtifact(f"{e}") from e


@st.cache_data(show_spinner="Running the model over the dataset ...")
def get_table(h: int):
    from src.predict import inference_table
    return inference_table(get_grid(), get_bundle(h))


@st.cache_data
def get_json(name: str, how: str = "Run `python -m src.train` and `python -m src.evaluate`."):
    p = C.METRICS_DIR / name
    _need(p, how)
    return json.loads(p.read_text())


@st.cache_data
def get_csv(name: str):
    p = C.METRICS_DIR / name
    _need(p, "Run `python -m src.evaluate`.")
    return pd.read_csv(p)


def guarded(page_fn):
    """Show a clear, actionable message instead of a stack trace."""
    def run():
        try:
            page_fn()
        except MissingArtifact as e:
            st.error(f"ParkSmart can't show this page yet. {e}")
            st.code("python -m src.data_preprocessing\npython -m src.eda\npython -m src.train\npython -m src.evaluate",
                    language="bash")
    run.__name__ = page_fn.__name__
    return run


def replay_note():
    st.markdown('<div class="ps-note"><b>Replay of historical data.</b> ParkSmart uses Birmingham City Council car-park '
                'counts from 4 Oct – 19 Dec 2016 (UCI dataset 482). Forecasts are model estimates made as if it were the '
                'chosen moment; they are not live sensor readings.</div>', unsafe_allow_html=True)


def fmt_time(h):
    return f"{int(h):02d}:{int(round((h % 1) * 60)):02d}"


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


# ----------------------------------------------------------------------------
# Page 1: Overview
# ----------------------------------------------------------------------------
@guarded
def page_overview():
    st.title("ParkSmart")
    st.markdown("#### Forecasting how full Birmingham's car parks will be in 30 and 60 minutes")
    replay_note()
    grid = get_grid()
    q = json.loads(C.QUALITY_JSON.read_text()) if C.QUALITY_JSON.exists() else {}
    obs = grid.dropna(subset=["occupancy_rate"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Car parks modelled", grid["SystemCodeNumber"].nunique(),
              help="30 in the raw data; 2 with under 500 readings were excluded.")
    c2.metric("Days of data", grid["date"].nunique(), help="4 Oct – 19 Dec 2016, 08:00–16:30 each day")
    c3.metric("Raw readings", f"{q.get('raw_audit', {}).get('rows', 0):,}")
    c4.metric("Total spaces", f"{int(grid.groupby('SystemCodeNumber')['capacity'].first().sum()):,}")
    c5.metric("Mean occupancy", pct(obs["occupancy_rate"].mean()))

    left, right = st.columns([3, 2])
    daily = obs.groupby("date")["occupancy_rate"].mean() * 100
    fig = go.Figure(go.Scatter(x=daily.index, y=daily.values, mode="lines+markers", line=dict(color=BLUE, width=2.5),
                               marker=dict(size=5), name="Daily mean"))
    for d, lab in ((C.TRAIN_END, "training ends"), (C.VAL_END, "test period starts")):
        fig.add_vline(x=pd.Timestamp(d) + pd.Timedelta(hours=12), line_dash="dash", line_color=MUTED)
        fig.add_annotation(x=pd.Timestamp(d), y=daily.max() + 2, text=lab, showarrow=False, font=dict(color=MUTED, size=11))
    fig.update_yaxes(title="Mean occupancy (%)")
    left.plotly_chart(fig_style(fig, "Daily mean occupancy, all car parks"), width="stretch")

    hp = obs.assign(hour=8 + obs["slot"] * 0.5).groupby("hour")["occupancy_rate"].mean() * 100
    fig = go.Figure(go.Bar(x=[fmt_time(h) for h in hp.index], y=hp.values, marker_color=BLUE))
    fig.update_yaxes(title="Mean occupancy (%)")
    right.plotly_chart(fig_style(fig, "Average day"), width="stretch")

    st.subheader("Car parks")
    fac = obs.groupby("SystemCodeNumber").agg(capacity=("capacity", "first"), mean=("occupancy_rate", "mean"),
                                              full=("status", lambda s: (s == "FULL").mean()),
                                              readings=("n_readings", "sum")).reset_index()
    fac = fac.sort_values("mean", ascending=False)
    st.dataframe(fac.rename(columns={"SystemCodeNumber": "Car park", "capacity": "Spaces", "mean": "Mean occupancy",
                                     "full": "Time spent ≥90% full", "readings": "Readings"}),
                 hide_index=True, width="stretch",
                 column_config={"Mean occupancy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                "Time spent ≥90% full": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    with st.expander("What is historical, what is predicted, what is live?"):
        st.markdown("""
- **Historical**: every occupancy figure on this page is a real count from the dataset.
- **Predicted**: the *Availability* and *Forecasting* pages show model outputs computed using only data up to the
  chosen moment (no peeking at the future).
- **Live**: nothing. The dataset ended in December 2016 and ParkSmart has no sensor, camera or API connection.
""")


# ----------------------------------------------------------------------------
# Page 2: Availability
# ----------------------------------------------------------------------------
@guarded
def page_availability():
    st.title("Will I find a space?")
    replay_note()
    grid = get_grid()
    parks = sorted(grid["SystemCodeNumber"].unique())
    c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.2, 1.2])
    park = c1.selectbox("Car park", parks, index=parks.index("Broad Street") if "Broad Street" in parks else 0)
    dates = sorted(grid["date"].dt.date.unique())
    test_start = pd.Timestamp(C.VAL_END).date()
    default_date = next((d for d in dates if d > test_start), dates[-1])
    date = c2.date_input("Date", value=default_date, min_value=dates[0], max_value=dates[-1],
                         help="Dates after 5 Dec 2016 were never seen while training or tuning.")
    horizon = c4.radio("Look ahead", [30, 60], index=1, format_func=lambda m: f"{m} min", horizontal=True)
    last = C.SLOTS_PER_DAY - 1 - C.HORIZONS[horizon]
    times = [fmt_time(8 + i * 0.5) for i in range(last + 1)]
    at = c3.selectbox("Current time", times, index=min(6, len(times) - 1))
    as_of = pd.Timestamp(f"{date} {at}")

    if pd.Timestamp(date) not in set(grid["date"]):
        st.warning("There is no data at all for this date (the source has gaps on 20–21 Oct and 3–4 Dec). Pick another day.")
        return
    bundle = get_bundle(horizon)
    table = get_table(horizon)
    from src.predict import forecast
    try:
        f = forecast(table, bundle, park, as_of)
    except ValueError as e:
        st.warning(str(e))
        return

    sign, detail = st.columns([1.15, 1])
    color = STATUS[f.predicted_status]
    sign.markdown(f"""
<div class="ps-vms">
  <div class="ps-vms-top"><span>{park}</span><span>Forecast for {f.forecast_for[-5:]}</span></div>
  <div class="ps-vms-num">{f.estimated_available:,}</div>
  <div class="ps-vms-unit">spaces expected free of {f.capacity:,}</div>
  <div class="ps-vms-status" style="background:{color}">{f.predicted_status} · {STATUS_TEXT[f.predicted_status]}</div>
  <div class="ps-vms-foot">Estimated by the model at {f.as_of[-5:]} on {as_of:%a %d %b %Y}, {horizon} minutes ahead.
  Not a live count.</div>
</div>""", unsafe_allow_html=True)

    lo, hi = f.interval_90
    free_lo, free_hi = max(0, round(f.capacity * (1 - hi))), max(0, round(f.capacity * (1 - lo)))
    detail.markdown(f"""
<div class="ps-kv">
<p>Occupancy now: <b>{pct(f.current_rate, 1)}</b> ({round(f.current_rate * f.capacity):,} cars)</p>
<p>Predicted occupancy at {f.forecast_for[-5:]}: <b>{pct(f.predicted_rate, 1)}</b>
 &nbsp;(90% range {pct(lo)}–{pct(hi)}, i.e. {free_lo:,}–{free_hi:,} free spaces)</p>
<p>Estimated available spaces = max(0, capacity − predicted occupied) = max(0, {f.capacity:,} − {f.estimated_occupied:,}) = <b>{f.estimated_available:,}</b></p>
</div>""", unsafe_allow_html=True)
    if f.actual_rate is not None:
        err = (f.actual_rate - f.predicted_rate) * 100
        detail.success(f"Replay check: the car park was actually **{pct(f.actual_rate, 1)}** full at {f.forecast_for[-5:]} "
                       f"({f.actual_status}); the forecast was off by **{err:+.1f} percentage points** "
                       f"(≈ {abs(err) / 100 * f.capacity:.0f} spaces).")
    else:
        detail.info("No reading exists in the data for the forecast time, so the forecast can't be checked.")
    detail.caption(f"Model: {f.model}. The 90% range comes from out-of-sample errors during walk-forward validation "
                   f"(±{bundle['interval_halfwidth_90'] * 100:.1f} pp).")

    st.subheader(f"All car parks at {as_of:%H:%M}, forecast for {f.forecast_for[-5:]}")
    snap = table[table["slot_time"] == as_of].copy()
    if snap.empty:
        st.info("No car park has a reading at this moment.")
        return
    snap = snap.sort_values("pred_available", ascending=False)
    snap["Status"] = snap["pred_status"]
    show = snap[["SystemCodeNumber", "capacity", "rate_now", "pred_rate", "pred_available", "Status", "actual_rate"]].rename(
        columns={"SystemCodeNumber": "Car park", "capacity": "Spaces", "rate_now": "Occupancy now",
                 "pred_rate": "Predicted occupancy", "pred_available": "Predicted free spaces", "actual_rate": "Actual (replay)"})
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                ("Occupancy now", "Predicted occupancy", "Actual (replay)")})
    alt = snap[(snap["SystemCodeNumber"] != park) & snap["pred_status"].isin(["LOW", "MODERATE"])]
    if f.predicted_status in ("HIGH", "FULL") and not alt.empty:
        st.markdown(f"**{park} is expected to be {f.predicted_status}.** Car parks forecast to have the most room: "
                    + ", ".join(f"{r.SystemCodeNumber} ({r.pred_available:,} free)" for r in alt.head(3).itertuples()) + ".")


# ----------------------------------------------------------------------------
# Page 3: Forecasting
# ----------------------------------------------------------------------------
@guarded
def page_forecasting():
    st.title("Forecasts through the day")
    replay_note()
    grid = get_grid()
    parks = sorted(grid["SystemCodeNumber"].unique())
    c1, c2 = st.columns([2, 1.3])
    park = c1.selectbox("Car park", parks, index=parks.index("Broad Street") if "Broad Street" in parks else 0, key="fc_park")
    dates = sorted(grid["date"].dt.date.unique())
    test_dates = [d for d in dates if d > pd.Timestamp(C.VAL_END).date()]
    date = c2.selectbox("Day (test period)", test_dates, index=len(test_dates) - 6, format_func=lambda d: d.strftime("%a %d %b"))

    actual = grid[(grid["SystemCodeNumber"] == park) & (grid["date"].dt.date == date)].sort_values("slot_time")
    fig = go.Figure()
    colors = {30: "#5B8FD6", 60: BLUE}
    for h in (60, 30):
        t = get_table(h)
        d = t[(t["SystemCodeNumber"] == park) & (t["date"].dt.date == date)].sort_values("target_time")
        if h == 60:
            fig.add_trace(go.Scatter(x=list(d["target_time"]) + list(d["target_time"])[::-1],
                                     y=list(d["pred_high"] * 100) + list(d["pred_low"] * 100)[::-1], fill="toself",
                                     fillcolor="rgba(11,79,156,0.12)", line=dict(width=0), name="60-min 90% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d["target_time"], y=d["pred_rate"] * 100, mode="lines+markers", name=f"Forecast made {h} min earlier",
                                 line=dict(color=colors[h], width=2, dash="dot" if h == 30 else "solid"), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=actual["slot_time"], y=actual["occupancy_rate"] * 100, mode="lines+markers", name="Actual",
                             line=dict(color=ASPHALT, width=3), marker=dict(size=6)))
    for lab, b in zip(C.STATUS_LABELS[1:], C.STATUS_BINS):
        fig.add_hline(y=b * 100, line_dash="dash", line_color=STATUS[lab], opacity=0.6,
                      annotation_text=f"{lab} ≥ {int(b * 100)}%", annotation_position="top left",
                      annotation_font_color=STATUS[lab])
    fig.update_yaxes(title="Occupancy (%)", range=[0, 105])
    st.plotly_chart(fig_style(fig, f"{park}, {date:%A %d %B %Y}", 430), width="stretch")

    t60 = get_table(60)
    day60 = t60[(t60["SystemCodeNumber"] == park) & (t60["date"].dt.date == date)].dropna(subset=["actual_rate"])
    if not day60.empty:
        mae = (day60["actual_rate"] - day60["pred_rate"]).abs().mean()
        st.caption(f"60-minute forecasts on this day: mean absolute error {mae * 100:.1f} percentage points "
                   f"≈ {mae * day60['capacity'].iloc[0]:.0f} spaces.")

    st.subheader("When is this car park usually busy?")
    hist = grid[(grid["SystemCodeNumber"] == park) & (grid["date"] <= C.VAL_END)].dropna(subset=["occupancy_rate"])
    hm = hist.assign(dow=hist["date"].dt.dayofweek, hour=8 + hist["slot"] * 0.5).groupby(["dow", "hour"])["occupancy_rate"].mean().unstack() * 100
    fig = go.Figure(go.Heatmap(z=hm.values, x=[fmt_time(h) for h in hm.columns], y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(hm)],
                               colorscale=[[0, "#EAF1FA"], [0.5, "#F2C230"], [0.8, "#E06A1B"], [1, "#C62828"]], zmin=0, zmax=100,
                               colorbar=dict(title="%")))
    st.plotly_chart(fig_style(fig, "Average occupancy, training + validation weeks", 320), width="stretch")
    wk = hist[hist["date"].dt.dayofweek < 5]["occupancy_rate"].mean()
    we = hist[hist["date"].dt.dayofweek >= 5]["occupancy_rate"].mean()
    st.caption(f"Weekday mean {pct(wk)} vs weekend mean {pct(we)}.")

    st.subheader("Where do forecasts go wrong?")
    eh = C.FIGURES_DIR / "eval_error_by_hour_h60.png"
    if eh.exists():
        st.image(str(eh), caption="Test-period error by time of day, 60 minutes ahead (from src/evaluate.py)")


# ----------------------------------------------------------------------------
# Page 4: Model performance
# ----------------------------------------------------------------------------
@guarded
def page_models():
    st.title("How good are the models?")
    ev = get_json("evaluation_summary.json")
    tr = get_json("training_summary.json")
    horizon = st.radio("Forecast horizon", [60, 30], format_func=lambda m: f"{m} minutes ahead", horizontal=True)
    e, t = ev[str(horizon)], tr[str(horizon)]
    reg = pd.DataFrame(e["regression_table"])
    sel = reg[reg["Selected"]].iloc[0]
    pers = reg[reg["Model"] == "Persistence (baseline)"].iloc[0]
    ws = reg[reg["Model"] == "Persistence + weekly shape (baseline)"].iloc[0]
    st.markdown(f"Selected model: **{sel['Model']}** (regression) · **{t['selected_classifier']}** (status)")
    c2, c3, c4 = st.columns(3)
    c2.metric("Test MAE", f"{sel['Test MAE (pp)']:.2f} pp", f"{sel['Test MAE (pp)'] - pers['Test MAE (pp)']:.2f} vs persistence",
              delta_color="inverse")
    c3.metric("≈ spaces off, on average", f"{sel['Test MAE (spaces)']:.0f}")
    c4.metric("90% range coverage on test", pct(e["interval_test_coverage_90"], 1),
              help="Share of test forecasts whose actual value fell inside the ±range shown in the app.")
    st.caption(f"Train {t['date_ranges']['train'][0]} → {t['date_ranges']['train'][1]} · validation "
               f"{t['date_ranges']['val'][0]} → {t['date_ranges']['val'][1]} · test {t['date_ranges']['test'][0]} → "
               f"{t['date_ranges']['test'][1]}. Models were chosen on walk-forward validation, never on test.")

    st.subheader("Predicting the occupancy rate (regression)")
    cols = ["Model", "Test MAE (pp)", "Test RMSE (pp)", "Test R2", "Test MAE (spaces)", "MAE vs persistence (%)",
            "Walk-forward MAE (pp)", "Walk-forward MAE std (pp)", "Fit time (s)", "Predict ms / 1000 rows", "Selected"]
    st.dataframe(reg[cols].round(3), hide_index=True, width="stretch")
    st.image(str(C.ROOT / e["figures"]["model_comparison"]))
    gain = (1 - sel["Test MAE (pp)"] / ws["Test MAE (pp)"]) * 100
    st.markdown(f"**Why {sel['Model']}?** It had the lowest walk-forward MAE "
                f"({sel['Walk-forward MAE (pp)']:.2f} pp). On the untouched test fortnight it is "
                f"{sel['MAE vs persistence (%)']:.0f}% more accurate than persistence and {gain:.0f}% more accurate than the "
                f"strong 'persistence + weekly shape' rule. {t['regression'][sel['Model']]['why']}")

    st.subheader("Predicting the status band (classification)")
    clf = pd.DataFrame(e["classification_table"])
    st.dataframe(clf[["Model", "accuracy", "macro_precision", "macro_recall", "macro_F1", "weighted_F1", "FULL_precision",
                      "FULL_recall", "HIGH_recall", "Walk-forward macro F1", "Selected"]].round(3),
                 hide_index=True, width="stretch")
    st.image(str(C.ROOT / e["figures"]["confusion"]))
    st.markdown(f"Of **{e['full_actual']}** test forecasts where the car park really was FULL, the selected classifier "
                f"missed **{e['full_missed']}**, and only **{e['full_missed_as_low_or_moderate']}** of those were called LOW or "
                f"MODERATE. It raised **{e['false_full_alarms']}** FULL warnings that turned out not to be FULL. "
                "Missing a FULL car park sends a driver somewhere they will circle and queue; a false FULL warning only "
                "nudges them to an alternative, so FULL recall matters most.")

    a, b = st.columns(2)
    a.image(str(C.ROOT / e["figures"]["actual_vs_pred"]))
    b.image(str(C.ROOT / e["figures"]["permutation_importance"]))
    imp = pd.DataFrame(e["permutation_importance"])
    top = imp.head(3)
    st.markdown("**What drives the forecast?** Shuffling a feature and measuring how much worse the model gets: "
                + "; ".join(f"`{r.feature}` (+{r.importance_MAE_increase_pp:.2f} pp)" for r in top.itertuples())
                + ". Importance shows what the model relies on, not what causes parking demand.")
    a, b = st.columns(2)
    a.image(str(C.ROOT / e["figures"]["per_park"]))
    b.image(str(C.ROOT / e["figures"]["error_by_hour"]))
    seg = pd.DataFrame(e["segments"]).T.reset_index().rename(columns={"index": "Segment"})
    st.dataframe(seg.round(3), hide_index=True, width="stretch")
    u = ev.get("unseen_park_experiment")
    if u and horizon == u["horizon_minutes"]:
        st.markdown(f"**Car parks the model has never seen:** with the car-park ID removed and each park held out in turn, "
                    f"test MAE is {u['unseen_park_MAE_pp']:.2f} pp (vs {u['seen_park_MAE_pp']:.2f} pp for known parks and "
                    f"{u['persistence_MAE_pp']:.2f} pp for persistence).")
    with st.expander("Representative test predictions"):
        st.dataframe(pd.DataFrame(e["prediction_examples"]), hide_index=True, width="stretch")
    with st.expander("Tuned hyper-parameters"):
        st.json({k: v["best_params"] for k, v in t["regression"].items()})


# ----------------------------------------------------------------------------
# Page 5: Insights
# ----------------------------------------------------------------------------
@guarded
def page_insights():
    st.title("What the data says")
    eda = get_json("eda_summary.json", "Run `python -m src.eda`.")
    ev = get_json("evaluation_summary.json")
    e = ev[str(C.PRIMARY_HORIZON)]
    fac = pd.DataFrame(eda["facilities"]).T
    full_parks = fac.sort_values("full_share", ascending=False)
    roomy = fac.sort_values("mean_rate")
    hp = pd.Series(eda["hour_profile_mean"])
    busy = hp[hp >= hp.max() - 0.03].index.tolist()
    g = get_grid().dropna(subset=["occupancy_rate"])
    wkend = g[g["date"].dt.dayofweek >= 5]
    we_oct = wkend[wkend["date"].dt.month == 10]["occupancy_rate"].mean()
    we_dec = wkend[wkend["date"].dt.month == 12]["occupancy_rate"].mean()

    st.subheader("For drivers")
    st.markdown(f"""
- **Busiest time of day:** demand peaks at **{eda['peak_time']}** (average {pct(eda['peak_mean_rate'])} full across
  car parks) and stays within 3 points of the peak from **{busy[0]} to {busy[-1]}**. Arriving before
  {busy[0]} is the easiest way to find space; 08:00 is the quietest slot.
- **Weekends are quieter on average** ({pct(eda['weekday_vs_weekend_mean']['weekend'])} vs
  {pct(eda['weekday_vs_weekend_mean']['weekday'])} on weekdays), but weekend demand grew towards Christmas: weekend
  mean {pct(we_oct)} in October vs {pct(we_dec)} in December.
- **Car parks most often full (≥90%):** """ + ", ".join(
        f"{p} ({pct(full_parks.loc[p, 'full_share'])} of slots)" for p in full_parks.index[:5]) + """.
- **Car parks that almost always have room:** """ + ", ".join(
        f"{p} (mean {pct(roomy.loc[p, 'mean_rate'])})" for p in roomy.index[:4]) + ".")

    st.subheader("For car-park operators")
    sel = e["selected_regressor"]
    worst = e["per_park_worst"]
    seg = e["segments"]
    pk, op = seg["peak (12:00-15:00)"]["selected_MAE_pp"], seg["off-peak"]["selected_MAE_pp"]
    wd, we = seg["weekday"]["selected_MAE_pp"], seg["weekend"]["selected_MAE_pp"]
    eh = pd.read_csv(C.PREDICTIONS_DIR / f"test_predictions_h{C.PRIMARY_HORIZON}.csv")
    eh["err"] = (eh["target_rate"] - eh[f"reg::{sel}"]).abs() * 100
    by_hour = eh.groupby(8 + eh["target_slot"] * 0.5)["err"].mean()
    hard_hours = ", ".join(fmt_time(h) for h in by_hour.sort_values(ascending=False).index[:3])
    harder_sentence = (
        f"Forecast error is {'higher' if op > pk else 'lower'} **off-peak** ({op:.2f} pp) than in the 12:00–15:00 peak "
        f"({pk:.2f} pp): the hardest forecast times are {hard_hours}, when car parks fill or empty fastest. Errors are "
        f"{'higher' if we > wd else 'lower'} on **weekends** ({we:.2f} pp) than weekdays ({wd:.2f} pp), so operators "
        "should keep a safety margin at those times.")
    st.markdown(f"""
- A **60-minute forecast** from {sel} is off by {pd.DataFrame(e['regression_table']).query('Selected')['Test MAE (spaces)'].iloc[0]:.0f}
  spaces on average, which is precise enough to switch "FULL" signs or open overflow areas before queues form.
- {harder_sentence}
- Car parks with the largest forecast errors: {', '.join(worst)}. These are candidates for better sensors or extra
  features (events, retail calendars).
- The model's 90% range covered {pct(e['interval_test_coverage_90'], 1)} of test outcomes, so the range shown to
  drivers is honest about uncertainty.
""")
    st.subheader("What ParkSmart cannot tell you")
    st.markdown("""
- It does not know about live events, roadworks, weather or prices: none are in the dataset.
- It cannot forecast overnight or before 08:00; the source only covers 08:00–16:30.
- A forecast needs a current reading for that car park; when a sensor drops out, no forecast is made.
- Results describe Birmingham in autumn 2016 and would need re-training before use anywhere else or today.
""")


pg = st.navigation([
    st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(page_availability, url_path="availability", title="Parking availability", icon=":material/local_parking:"),
    st.Page(page_forecasting, url_path="forecasting", title="Forecasting", icon=":material/timeline:"),
    st.Page(page_models, url_path="models", title="Model performance", icon=":material/analytics:"),
    st.Page(page_insights, url_path="insights", title="Insights", icon=":material/lightbulb:"),
])
with st.sidebar:
    st.markdown('<div class="ps-brand"><div class="ps-p">P</div><div><div class="ps-name">ParkSmart</div>'
                '<div class="ps-sub">Birmingham, Oct–Dec 2016</div></div></div>', unsafe_allow_html=True)
    st.caption("AUCML ESE project · Symbiosis Institute of Technology, Pune")
pg.run()
