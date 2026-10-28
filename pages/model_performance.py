import pandas as pd
import streamlit as st

from src import config as C
from src.ui import birmingham_notice, guarded, metrics_json, page_header
from src.visualizations import (actual_vs_predicted, confusion, error_by_hour, importance, interval_coverage,
                                model_comparison, per_park)


@st.cache_data
def test_predictions(h: int) -> pd.DataFrame:
    return pd.read_csv(C.PREDICTIONS_DIR / f"test_predictions_h{h}.csv")


def chart(fig):
    st.plotly_chart(fig, width="stretch", theme="streamlit", config={"displayModeBar": False})


def page():
    page_header("Model performance", "Every number here was produced by `src/train.py` and `src/evaluate.py` on the "
                                     "Birmingham data and is re-checked by the test suite.", "model")
    birmingham_notice()
    ev = metrics_json("evaluation_summary.json")
    tr = metrics_json("training_summary.json")
    horizon = st.segmented_control("Forecast horizon", [60, 30], default=60, required=True,
                                   format_func=lambda m: f"{m} minutes ahead")
    e, t = ev[str(horizon)], tr[str(horizon)]
    reg = pd.DataFrame(e["regression_table"])
    sel = reg[reg["Selected"]].iloc[0]
    pers = reg[reg["Model"] == "Persistence (baseline)"].iloc[0]
    rule = reg[reg["Model"] == "Persistence + weekly shape (baseline)"].iloc[0]
    clf = pd.DataFrame(e["classification_table"])
    sel_c = clf[clf["Selected"]].iloc[0]

    m = st.columns(4)
    m[0].metric("Test error (MAE)", f"{sel['Test MAE (pp)']:.2f} pp",
                f"{sel['Test MAE (pp)'] - pers['Test MAE (pp)']:.2f} pp vs persistence", delta_color="inverse")
    m[1].metric("≈ spaces off on average", f"{sel['Test MAE (spaces)']:.0f}")
    m[2].metric("Status accuracy", f"{sel_c['accuracy']:.1%}", help=f"Macro-F1 {sel_c['macro_F1']:.3f}")
    band = e.get("interval_band_report")
    m[3].metric("Inside the 90% range", f"{(band['band_coverage'] if band else e['interval_test_coverage_90']):.1%}")
    st.caption(f"Selected: **{sel['Model']}** for occupancy, **{t['selected_classifier']}** for status. "
               f"Train {t['date_ranges']['train'][0]} → {t['date_ranges']['train'][1]}, validation "
               f"{t['date_ranges']['val'][0]} → {t['date_ranges']['val'][1]}, test {t['date_ranges']['test'][0]} → "
               f"{t['date_ranges']['test'][1]}. Models were chosen on walk-forward validation, never on test.")

    st.subheader("Occupancy-rate forecasts")
    a, b = st.columns([1.25, 1], gap="large")
    with a:
        chart(model_comparison(reg))
    with b:
        gain = (1 - sel["Test MAE (pp)"] / rule["Test MAE (pp)"]) * 100
        st.markdown(f"**Why {sel['Model']}?** Lowest walk-forward error ({sel['Walk-forward MAE (pp)']:.2f} pp), which "
                    f"is how it was chosen. On the untouched test fortnight it is "
                    f"{sel['MAE vs persistence (%)']:.0f}% better than persistence and {gain:.0f}% better than the "
                    "hand-made *persistence + last week's shape* rule.")
        st.markdown("Ridge regression does **not** beat that rule, so the gain comes from the non-linear trees, not "
                    "from using machine learning as such.")
        st.caption(t["regression"][sel["Model"]]["why"])
    cols = ["Model", "Test MAE (pp)", "Test RMSE (pp)", "Test R2", "Test MAE (spaces)", "MAE vs persistence (%)",
            "Walk-forward MAE (pp)", "Fit time (s)", "Predict ms / 1000 rows"]
    st.dataframe(reg[cols].round(3), hide_index=True, width="stretch")

    tp = test_predictions(horizon)
    sel_col = f"reg::{sel['Model']}"
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Actual vs predicted (test)**")
        chart(actual_vs_predicted(tp, sel_col))
    with b:
        st.markdown("**Error by time of day**")
        chart(error_by_hour(tp, sel_col))
        seg = e["segments"]
        st.caption(f"Peak 12:00–15:00: {seg['peak (12:00-15:00)']['selected_MAE_pp']:.2f} pp · off-peak "
                   f"{seg['off-peak']['selected_MAE_pp']:.2f} pp · weekdays {seg['weekday']['selected_MAE_pp']:.2f} pp · "
                   f"weekends {seg['weekend']['selected_MAE_pp']:.2f} pp.")

    st.subheader("Status forecasts (LOW / MODERATE / HIGH / FULL)")
    st.dataframe(clf[["Model", "accuracy", "macro_precision", "macro_recall", "macro_F1", "weighted_F1",
                      "FULL_precision", "FULL_recall", "HIGH_recall", "Walk-forward macro F1"]].round(3),
                 hide_index=True, width="stretch")
    a, b = st.columns(2, gap="large")
    cms = e["confusion_matrices"]
    with a:
        chart(confusion(cms["Persistence (baseline)"], C.STATUS_LABELS, "Persistence"))
    with b:
        chart(confusion(cms[t["selected_classifier"]], C.STATUS_LABELS, t["selected_classifier"]))
    st.markdown(f"Of **{e['full_actual']}** test cases that really were FULL, the selected model missed "
                f"**{e['full_missed']}**, and only **{e['full_missed_as_low_or_moderate']}** were called LOW or MODERATE. "
                f"It gave **{e['false_full_alarms']}** FULL warnings that weren't. Missing a full car park sends a driver "
                "into a queue; a false warning only suggests an alternative.")

    st.subheader("Uncertainty")
    if band:
        a, b = st.columns([1.2, 1], gap="large")
        with a:
            chart(interval_coverage(band))
        with b:
            st.markdown("The 90% range is built from walk-forward errors on earlier weeks (split-conformal). One width "
                        "for all times was too narrow in the morning and too wide in the afternoon, so ParkSmart uses a "
                        "width per time-of-day band.")
            st.dataframe(pd.DataFrame([{"Time band": k, "Half-width (pp)": v["band_halfwidth_pp"],
                                        "Coverage, one width": v["global_coverage"], "Coverage, by band": v["band_coverage"]}
                                       for k, v in band["by_band"].items()]).round(3), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="percent")
                                        for c in ("Coverage, one width", "Coverage, by band")})

    st.subheader("What the model relies on")
    a, b = st.columns([1.2, 1], gap="large")
    imp = pd.DataFrame(e["permutation_importance"])
    with a:
        chart(importance(imp))
    with b:
        top = imp.head(4)
        st.markdown("Permutation importance: how much worse the test error gets when one feature is shuffled. "
                    + "; ".join(f"`{r.feature}` +{r.importance_MAE_increase_pp:.2f} pp" for r in top.itertuples()) + ".")
        st.markdown("`target_hour` and `target_slot` look unimportant only because `origin_slot` carries the same "
                    "information. Importance describes the model, not what causes parking demand.")

    st.subheader("Generalisation")
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        pp = pd.read_csv(C.METRICS_DIR / f"per_park_errors_h{horizon}.csv")
        chart(per_park(pp))
    with b:
        u = ev.get("unseen_park_experiment")
        if u and horizon == u["horizon_minutes"]:
            st.markdown(f"**Car parks the model never saw.** With the car-park ID removed and each car park held out in "
                        f"turn: {u['unseen_park_MAE_pp']:.2f} pp, against {u['seen_park_MAE_pp']:.2f} pp for known car "
                        f"parks and {u['persistence_MAE_pp']:.2f} pp for persistence.")
        try:
            ts = metrics_json("transfer_summary.json", "Run `python -m src.transfer`.")[str(horizon)]
            st.markdown(f"**Transfer model** (used by *Test on your own data*): on the held-out car park "
                        f"{ts['holdout_park']}, {ts['holdout_MAE_pp']:.2f} pp vs {ts['holdout_persistence_MAE_pp']:.2f} pp "
                        f"for persistence; {ts['holdout_interval_coverage']:.0%} inside its 90% range.")
        except Exception:
            pass
        st.markdown("**What this does not show:** that the model works in Pune. That needs Pune occupancy counts, "
                    "which are not publicly available.")
    with st.expander("Tuned hyper-parameters and representative predictions"):
        st.json({k: v["best_params"] for k, v in t["regression"].items()})
        st.dataframe(pd.DataFrame(e["prediction_examples"]), hide_index=True, width="stretch")


guarded(page)
