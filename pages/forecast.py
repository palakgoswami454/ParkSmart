import pandas as pd
import streamlit as st

from src import config as C
from src.ui import birmingham_notice, bundle, grid, guarded, inference, metrics_json, page_header
from src.visualizations import fmt_time, forecast_day, weekly_heatmap

OCC_BADGE = {"LOW": "green", "MODERATE": "yellow", "HIGH": "orange", "FULL": "red"}
OCC_TEXT = {"LOW": "plenty of space", "MODERATE": "filling up", "HIGH": "busy", "FULL": "full or nearly full"}


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def replay_tab():
    g = grid()
    parks = sorted(g["SystemCodeNumber"].unique())
    c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.2, 1.3])
    park = c1.selectbox("Car park (Birmingham)", parks, index=parks.index("Broad Street"))
    dates = sorted(g["date"].dt.date.unique())
    default = next(d for d in dates if d > pd.Timestamp(C.VAL_END).date())
    day = c2.date_input("Date", value=default, min_value=dates[0], max_value=dates[-1],
                        help="Dates after 5 Dec 2016 were never used for training or tuning.")
    horizon = c4.segmented_control("Look ahead", [30, 60], default=60, format_func=lambda m: f"{m} min", required=True)
    last = C.SLOTS_PER_DAY - 1 - C.HORIZONS[horizon]
    times = [fmt_time(8 + i * 0.5) for i in range(last + 1)]
    at = c3.selectbox("Current time", times, index=min(6, len(times) - 1))
    as_of = pd.Timestamp(f"{day} {at}")
    if pd.Timestamp(day) not in set(g["date"]):
        st.warning("The source has no data for this date (gaps on 20–21 Oct and 3–4 Dec 2016). Pick another day.",
                   icon=":material/event_busy:")
        return
    from src.predict import forecast
    try:
        f = forecast(inference(horizon), bundle(horizon), park, as_of)
    except ValueError as e:
        st.warning(str(e), icon=":material/sensors_off:")
        return

    left, right = st.columns([1, 1.15], gap="large")
    with left, st.container(border=True):
        st.caption(f"{park} · forecast for {f.forecast_for[-5:]}, made at {f.as_of[-5:]} on {as_of:%a %d %b %Y}")
        st.metric("Spaces expected free", f"{f.estimated_available:,}", help="max(0, capacity − predicted occupied)")
        st.markdown(f"of **{f.capacity:,}** spaces")
        st.badge(f"{f.predicted_status} · {OCC_TEXT[f.predicted_status]}", color=OCC_BADGE[f.predicted_status])
        st.caption("Model estimate for a replayed moment in 2016. Not a live count.")
    lo, hi = f.interval_90
    with right:
        st.markdown(f"Occupancy at {f.as_of[-5:]}: **{pct(f.current_rate, 1)}** ({round(f.current_rate * f.capacity):,} cars)")
        st.markdown(f"Predicted at {f.forecast_for[-5:]}: **{pct(f.predicted_rate, 1)}**, 90% range {pct(lo)}–{pct(hi)} "
                    f"(≈ {max(0, round(f.capacity * (1 - hi))):,}–{max(0, round(f.capacity * (1 - lo))):,} free)")
        st.markdown(f"Free spaces = max(0, {f.capacity:,} − {f.estimated_occupied:,}) = **{f.estimated_available:,}**")
        if f.actual_rate is not None:
            err = (f.actual_rate - f.predicted_rate) * 100
            st.success(f"**Replay check:** actually {pct(f.actual_rate, 1)} full ({f.actual_status}). The forecast was off "
                       f"by {err:+.1f} percentage points (≈ {abs(err) / 100 * f.capacity:.0f} spaces).",
                       icon=":material/fact_check:")
        else:
            st.info("There is no reading at the forecast time, so this forecast can't be checked.")
        st.caption(f"Model: {f.model}. The range width depends on the time of day and comes from walk-forward errors "
                   "on earlier weeks.")

    snap = inference(horizon)
    snap = snap[snap["slot_time"] == as_of].sort_values("pred_available", ascending=False)
    if not snap.empty:
        st.markdown(f"**All Birmingham car parks at {as_of:%H:%M}, forecast for {f.forecast_for[-5:]}**")
        st.dataframe(snap[["SystemCodeNumber", "capacity", "rate_now", "pred_rate", "pred_available", "pred_status",
                           "actual_rate"]].rename(columns={
            "SystemCodeNumber": "Car park", "capacity": "Spaces", "rate_now": "Now", "pred_rate": "Predicted",
            "pred_available": "Predicted free", "pred_status": "Status", "actual_rate": "Actual (replay)"}),
            hide_index=True, width="stretch", height=300,
            column_config={c: st.column_config.NumberColumn(format="percent") for c in ("Now", "Predicted", "Actual (replay)")})


def day_tab():
    g = grid()
    parks = sorted(g["SystemCodeNumber"].unique())
    c1, c2 = st.columns([2, 1.3])
    park = c1.selectbox("Car park (Birmingham)", parks, index=parks.index("Broad Street"), key="day_park")
    test_dates = [d for d in sorted(g["date"].dt.date.unique()) if d > pd.Timestamp(C.VAL_END).date()]
    day = c2.selectbox("Day in the test fortnight", test_dates, index=len(test_dates) - 6,
                       format_func=lambda d: d.strftime("%a %d %b %Y"))
    actual = g[(g["SystemCodeNumber"] == park) & (g["date"].dt.date == day)].sort_values("slot_time")
    st.plotly_chart(forecast_day(actual, {h: inference(h) for h in (30, 60)}, park, day), width="stretch",
                    theme="streamlit", config={"displayModeBar": False})
    t60 = inference(60)
    d60 = t60[(t60["SystemCodeNumber"] == park) & (t60["date"].dt.date == day)].dropna(subset=["actual_rate"])
    if not d60.empty:
        mae = (d60["actual_rate"] - d60["pred_rate"]).abs().mean()
        cov = ((d60["actual_rate"] - d60["pred_rate"]).abs() <= d60["interval_halfwidth"]).mean()
        st.caption(f"60-minute forecasts this day: mean error {mae * 100:.1f} pp ≈ {mae * d60['capacity'].iloc[0]:.0f} "
                   f"spaces; {cov:.0%} of outcomes inside the 90% range.")
    st.markdown("**Usual weekly pattern** (training and validation weeks only)")
    hist = g[(g["SystemCodeNumber"] == park) & (g["date"] <= C.VAL_END)].dropna(subset=["occupancy_rate"])
    st.plotly_chart(weekly_heatmap(hist), width="stretch", theme="streamlit", config={"displayModeBar": False})


def upload_tab():
    from src.transfer import SAMPLE_CSV, SUMMARY_JSON, UploadError, evaluate_upload, model_path
    st.markdown("Have occupancy counts for a car park, for example from a Pune operator? Upload them to check whether "
                "the model works there **before** trusting its forecasts. ParkSmart uses a version of the model that "
                "doesn't know car-park identities, measures its error on your data, and compares it with simply "
                "assuming occupancy stays the same.")
    st.markdown("CSV columns: `facility`, `capacity`, `occupied`, `timestamp` (the Birmingham names `SystemCodeNumber`, "
                "`Capacity`, `Occupancy`, `LastUpdated` also work). Readings about every 30 minutes, 08:00–16:30, "
                "at least 3 days.")
    if not model_path(60).exists():
        st.error("The transfer model hasn't been trained. Run `python -m src.transfer`.")
        return
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    up = c1.file_uploader("Occupancy CSV", type=["csv"])
    use_sample = c2.button("Try the sample file", icon=":material/science:",
                           help="A Birmingham car park that the transfer model never saw in training.")
    horizon = st.segmented_control("Forecast horizon", [30, 60], default=60, format_func=lambda m: f"{m} min",
                                   required=True, key="up_h")
    if use_sample:
        st.session_state["upload_df"], st.session_state["upload_name"] = pd.read_csv(SAMPLE_CSV), "sample (Birmingham, unseen car park)"
    elif up is not None:
        try:
            st.session_state["upload_df"], st.session_state["upload_name"] = pd.read_csv(up), up.name
        except Exception as e:  # malformed CSV
            st.error(f"Couldn't read that file as CSV: {e}")
            return
    df = st.session_state.get("upload_df")
    if df is None:
        if SUMMARY_JSON.exists():
            s = metrics_json("transfer_summary.json", "Run `python -m src.transfer`.")[str(horizon)]
            st.caption(f"Held-out check: on {s['holdout_park']}, which it never saw, the transfer model's {horizon}-min "
                       f"error was {s['holdout_MAE_pp']:.2f} pp vs {s['holdout_persistence_MAE_pp']:.2f} pp for "
                       "persistence. That is one car park in one city; your data may differ.")
        return
    try:
        res = evaluate_upload(df, horizon)
    except UploadError as e:
        st.error(str(e), icon=":material/error:")
        return
    st.markdown(f"**{st.session_state['upload_name']}** · {res.n_rows_used:,} readings used · {len(res.facilities)} "
                f"car park(s) · {res.days} days")
    for n in res.notes:
        st.caption(n)
    if res.model_mae_pp is None:
        st.warning("None of the forecasts could be checked against a later reading, so accuracy is unknown. "
                   "Forecasts are not shown.")
        return
    m = st.columns(3)
    m[0].metric("Model error on your data", f"{res.model_mae_pp:.2f} pp")
    m[1].metric("Persistence error", f"{res.persistence_mae_pp:.2f} pp",
                help="Error of assuming occupancy stays the same for the next 30/60 minutes.")
    m[2].metric("Inside the 90% range", f"{res.coverage:.0%}")
    if res.beats_persistence:
        st.success(f"The model beats persistence on your data ({res.n_scored:,} checked forecasts), so its forecasts "
                   "below are worth considering. Keep checking as more data arrives.", icon=":material/check_circle:")
        last = res.latest.rename(columns={"SystemCodeNumber": "Car park", "slot_time": "Last reading", "target_time":
                                          "Forecast for", "capacity": "Spaces", "rate_now": "Occupancy then",
                                          "pred": "Predicted", "pred_free": "Predicted free"})
        st.dataframe(last[["Car park", "Last reading", "Forecast for", "Spaces", "Occupancy then", "Predicted",
                           "Predicted free"]], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in ("Occupancy then", "Predicted")})
    else:
        st.error("The model does **not** beat persistence on your data, so ParkSmart won't show its forecasts for it. "
                 "It would need retraining on data from these car parks.", icon=":material/block:")
    if not res.per_facility.empty:
        with st.expander("Error by car park"):
            st.dataframe(res.per_facility.round(2), hide_index=True, width="stretch")


def page():
    page_header("Occupancy forecast", "How ParkSmart forecasts occupancy 30 and 60 minutes ahead, demonstrated on the "
                                      "data it was trained on.", "model")
    birmingham_notice()
    t1, t2, t3 = st.tabs([":material/history: Replay a forecast", ":material/timeline: Through the day",
                          ":material/upload_file: Test on your own data"])
    with t1:
        replay_tab()
    with t2:
        day_tab()
    with t3:
        upload_tab()


guarded(page)
