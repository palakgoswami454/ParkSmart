"""Evaluate saved test-period predictions and produce tables + figures.

Requires ``python -m src.train`` to have been run.
Run:  python -m src.evaluate
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score)
from sklearn.model_selection import GroupKFold

from src import config as C
from src import plot_style as S
from src.data_preprocessing import load_grid
from src.feature_engineering import FEATURES, build_features, park_codes, split_by_date
from src.intervals import coverage_report, width_for
from src.models import regression_candidates

S.apply()
PEAK_HOURS = (12.0, 15.0)  # target clock time 12:00-15:00 (window around the 14:00 maximum seen in EDA)


def _save(fig, name):
    fig.savefig(C.FIGURES_DIR / name)
    plt.close(fig)
    return f"outputs/figures/{name}"


def reg_metrics(y, p, cap=None):
    p = np.clip(p, 0, 1)
    out = {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)}
    if cap is not None:
        out["MAE_spaces"] = float(np.mean(np.abs(y - p) * cap))
    return out


def clf_metrics(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=C.STATUS_LABELS, zero_division=0)
    out = {"accuracy": accuracy_score(y, p),
           "macro_precision": float(np.mean(pr)), "macro_recall": float(np.mean(rc)),
           "macro_F1": f1_score(y, p, average="macro", labels=C.STATUS_LABELS, zero_division=0),
           "weighted_F1": f1_score(y, p, average="weighted", labels=C.STATUS_LABELS, zero_division=0)}
    for lab, a, b, c in zip(C.STATUS_LABELS, pr, rc, f1):
        out[f"{lab}_precision"], out[f"{lab}_recall"], out[f"{lab}_F1"] = a, b, c
    return out


def evaluate_horizon(h: int, summary: dict, grid: pd.DataFrame) -> dict:
    tp = pd.read_csv(C.PREDICTIONS_DIR / f"test_predictions_h{h}.csv", parse_dates=["date", "slot_time", "target_time"])
    s = summary[str(h)]
    res: dict = {"figures": {}}
    y, cap = tp["target_rate"].to_numpy(), tp["capacity"].to_numpy()
    tp["target_hour"] = 8 + tp["target_slot"] * 0.5

    # ---------------- Regression comparison table -------------------------
    rows = []
    pers_mae = mean_absolute_error(y, tp["reg::Persistence (baseline)"])
    for name, info in s["regression"].items():
        m = reg_metrics(y, tp[f"reg::{name}"].to_numpy(), cap)
        rows.append({"Model": name, "Test MAE (pp)": m["MAE"] * 100, "Test RMSE (pp)": m["RMSE"] * 100,
                     "Test R2": m["R2"], "Test MAE (spaces)": m["MAE_spaces"],
                     "MAE vs persistence (%)": (1 - m["MAE"] / pers_mae) * 100,
                     "Walk-forward MAE (pp)": info["walk_forward_MAE_mean"] * 100,
                     "Walk-forward MAE std (pp)": info["walk_forward_MAE_std"] * 100,
                     "Validation MAE (pp)": info["validation_MAE"] * 100,
                     "Fit time (s)": info["final_fit_seconds"],
                     "Predict ms / 1000 rows": info["predict_ms_per_1000_rows"],
                     "Selected": name == s["selected_regressor"]})
    reg_tab = pd.DataFrame(rows)
    reg_tab.round(4).to_csv(C.METRICS_DIR / f"regression_comparison_h{h}.csv", index=False)
    res["regression_table"] = reg_tab.round(4).to_dict(orient="records")

    # ---------------- Classification comparison table ---------------------
    rows = []
    for name, info in s["classification"].items():
        m = clf_metrics(tp["target_status"], tp[f"clf::{name}"])
        rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()},
                     "Walk-forward macro F1": info["walk_forward_macro_F1_mean"],
                     "Walk-forward macro F1 std": info["walk_forward_macro_F1_std"],
                     "Selected": name == s["selected_classifier"]})
    clf_tab = pd.DataFrame(rows)
    clf_tab.to_csv(C.METRICS_DIR / f"classification_comparison_h{h}.csv", index=False)
    res["classification_table"] = clf_tab.to_dict(orient="records")

    sel_r, sel_c = s["selected_regressor"], s["selected_classifier"]
    pred = tp[f"reg::{sel_r}"].to_numpy()
    tp["pred"] = pred
    tp["abs_err"] = np.abs(y - pred)

    # ---------------- Prediction interval coverage ------------------------
    hw = s["interval"]["halfwidth_90"]
    res["interval_test_coverage_90"] = float(np.mean(tp["abs_err"] <= hw))
    res["interval_halfwidth_90_pp"] = hw * 100
    table = joblib.load(C.MODELS_DIR / f"parksmart_h{h}.joblib").get("interval_table_90")
    if table:
        res["interval_band_report"] = coverage_report(y, pred, tp["target_hour"], table)
        tp["hw"] = width_for(tp["target_hour"], table)
    else:
        tp["hw"] = hw

    # ---------------- Segment analysis -----------------------------------
    peak = tp["target_hour"].between(*PEAK_HOURS)
    weekend = tp["day_of_week"] >= 5
    seg = {}
    for lab, mask in {"peak (12:00-15:00)": peak, "off-peak": ~peak, "weekday": ~weekend, "weekend": weekend}.items():
        seg[lab] = {"n": int(mask.sum()),
                    "selected_MAE_pp": float(tp.loc[mask, "abs_err"].mean() * 100),
                    "persistence_MAE_pp": float(np.abs(y[mask] - tp.loc[mask, "reg::Persistence (baseline)"]).mean() * 100),
                    "weekly_shape_MAE_pp": float(np.abs(y[mask] - tp.loc[mask, "reg::Persistence + weekly shape (baseline)"]).mean() * 100)}
    res["segments"] = seg

    hour_err = tp.groupby("target_hour").apply(lambda d: pd.Series({
        "selected": d["abs_err"].mean() * 100,
        "persistence": np.abs(d["target_rate"] - d["reg::Persistence (baseline)"]).mean() * 100,
        "weekly_shape": np.abs(d["target_rate"] - d["reg::Persistence + weekly shape (baseline)"]).mean() * 100}),
        include_groups=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(hour_err.index, hour_err["persistence"], color=S.MUTED, marker="o", ms=3, label="Persistence")
    ax.plot(hour_err.index, hour_err["weekly_shape"], color=S.ACCENT2, marker="o", ms=3, label="Persistence + weekly shape")
    ax.plot(hour_err.index, hour_err["selected"], color=S.ACCENT, marker="o", ms=3, lw=2.2, label=sel_r)
    ax.axvspan(*PEAK_HOURS, color=S.ACCENT, alpha=0.06)
    ax.set(title=f"Test error by forecast time ({h} min ahead)", xlabel="Target time of day (h)", ylabel="MAE (percentage points)")
    ax.legend()
    res["figures"]["error_by_hour"] = _save(fig, f"eval_error_by_hour_h{h}.png")

    park_err = tp.groupby("SystemCodeNumber").apply(lambda d: pd.Series({
        "MAE_pp": d["abs_err"].mean() * 100, "MAE_spaces": (d["abs_err"] * d["capacity"]).mean(),
        "persistence_MAE_pp": np.abs(d["target_rate"] - d["reg::Persistence (baseline)"]).mean() * 100,
        "capacity": d["capacity"].iloc[0], "n": len(d)}), include_groups=False).sort_values("MAE_pp")
    park_err.round(3).to_csv(C.METRICS_DIR / f"per_park_errors_h{h}.csv")
    res["per_park_best"] = park_err.head(3).index.tolist()
    res["per_park_worst"] = park_err.tail(3).index.tolist()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    yy = np.arange(len(park_err))
    ax.barh(yy + 0.2, park_err["persistence_MAE_pp"], height=0.4, color=S.MUTED, alpha=0.6, label="Persistence")
    ax.barh(yy - 0.2, park_err["MAE_pp"], height=0.4, color=S.ACCENT, label=sel_r)
    ax.set_yticks(yy, park_err.index, fontsize=8)
    ax.set(title=f"Test MAE per car park ({h} min ahead)", xlabel="MAE (percentage points)")
    ax.legend(loc="lower right")
    res["figures"]["per_park"] = _save(fig, f"eval_per_park_mae_h{h}.png")

    # ---------------- Actual vs predicted + residuals ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y * 100, pred * 100, s=4, alpha=0.25, color=S.ACCENT)
    axes[0].plot([0, 100], [0, 100], color=S.INK, lw=1, ls="--")
    axes[0].set(title="Actual vs predicted (test)", xlabel="Actual occupancy (%)", ylabel="Predicted occupancy (%)")
    resid = (y - pred) * 100
    axes[1].hist(resid, bins=80, color=S.ACCENT, alpha=0.85)
    axes[1].axvline(0, color=S.INK, lw=1)
    axes[1].set(title="Residuals (actual - predicted)", xlabel="Percentage points", ylabel="Count")
    fig.suptitle(f"{sel_r}, {h} min ahead", fontsize=11, fontweight="bold")
    res["figures"]["actual_vs_pred"] = _save(fig, f"eval_actual_vs_pred_h{h}.png")
    res["residual_mean_pp"] = float(resid.mean())
    res["residual_p95_abs_pp"] = float(np.percentile(np.abs(resid), 95))

    # ---------------- Model comparison bar chart -------------------------
    order = reg_tab.sort_values("Test MAE (pp)", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cols = [S.ACCENT if r else S.MUTED for r in order["Selected"]]
    ax.barh(order["Model"], order["Test MAE (pp)"], color=cols, alpha=0.9, label="Test")
    ax.errorbar(order["Walk-forward MAE (pp)"], order["Model"], xerr=order["Walk-forward MAE std (pp)"],
                fmt="o", color=S.ACCENT2, ms=4, label="Walk-forward mean ± std")
    ax.set(title=f"Regression models, {h} min ahead (lower is better)", xlabel="MAE (percentage points)")
    ax.legend(loc="lower right")
    res["figures"]["model_comparison"] = _save(fig, f"eval_model_comparison_h{h}.png")

    # ---------------- Confusion matrices ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.subplots_adjust(wspace=0.45)
    for ax, name in zip(axes, ["Persistence (baseline)", sel_c]):
        cm = confusion_matrix(tp["target_status"], tp[f"clf::{name}"], labels=C.STATUS_LABELS)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.6 else S.INK)
        ax.set_xticks(range(4), C.STATUS_LABELS)
        ax.set_yticks(range(4), C.STATUS_LABELS)
        ax.grid(False)
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
        res.setdefault("confusion_matrices", {})[name] = cm.tolist()
    fig.suptitle(f"Occupancy-status confusion matrices, {h} min ahead (row-normalised)", fontsize=11, fontweight="bold")
    res["figures"]["confusion"] = _save(fig, f"eval_confusion_h{h}.png")

    # FULL-related error analysis (the costly mistakes)
    full = tp["target_status"] == "FULL"
    sel_status = tp[f"clf::{sel_c}"]
    res["full_missed"] = int((full & (sel_status != "FULL")).sum())
    res["full_missed_as_low_or_moderate"] = int((full & sel_status.isin(["LOW", "MODERATE"])).sum())
    res["full_actual"] = int(full.sum())
    res["false_full_alarms"] = int((~full & (sel_status == "FULL")).sum())

    # ---------------- One example day ------------------------------------
    bundle = joblib.load(C.MODELS_DIR / f"parksmart_h{h}.joblib")
    ex_park = "Broad Street"
    ex = tp[tp["SystemCodeNumber"] == ex_park]
    ex_day = ex.groupby("date")["abs_err"].size().idxmax()
    exd = ex[ex["date"] == ex_day].sort_values("target_time")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    th = exd["target_time"].dt.hour + exd["target_time"].dt.minute / 60
    ax.plot(th, exd["target_rate"] * 100, color=S.INK, lw=2.2, marker="o", ms=4, label="Actual")
    ax.plot(th, exd["reg::Persistence (baseline)"] * 100, color=S.MUTED, ls="--", label="Persistence")
    ax.plot(th, exd["pred"] * 100, color=S.ACCENT, lw=2, marker="s", ms=4, label=f"{sel_r}")
    ax.fill_between(th, (exd["pred"] - exd["hw"]).clip(0) * 100, (exd["pred"] + exd["hw"]).clip(upper=1) * 100,
                    color=S.ACCENT, alpha=0.15, label="90% range (by time of day)")
    ax.set(title=f"{ex_park}, {pd.Timestamp(ex_day).date()} - forecasts made {h} min earlier",
           xlabel="Time of day (h)", ylabel="Occupancy (%)")
    ax.legend(fontsize=8)
    res["figures"]["example_day"] = _save(fig, f"eval_example_day_h{h}.png")

    # Representative prediction examples (quantiles of error)
    srt = tp.sort_values("abs_err").reset_index(drop=True)
    picks = [int(q * (len(srt) - 1)) for q in (0.1, 0.5, 0.9, 0.99)]
    ex_rows = srt.loc[picks, ["SystemCodeNumber", "slot_time", "target_time", "capacity", "rate_now", "target_rate", "pred",
                              "target_status", f"clf::{sel_c}"]].copy()
    ex_rows["error_pp"] = (ex_rows["target_rate"] - ex_rows["pred"]) * 100
    ex_rows["error_quantile"] = ["10th pct", "median", "90th pct", "99th pct"]
    ex_rows.round({"capacity": 0, "rate_now": 4, "target_rate": 4, "pred": 4, "error_pp": 2}).to_csv(C.METRICS_DIR / f"prediction_examples_h{h}.csv", index=False)
    res["prediction_examples"] = ex_rows.astype(str).to_dict(orient="records")

    # ---------------- Permutation importance -----------------------------
    codes = bundle["park_codes"]
    feats = build_features(grid, h, codes=codes)
    test = split_by_date(feats)["test"]
    t0 = time.perf_counter()
    pi = permutation_importance(bundle["regressor"], test[FEATURES], test["target_rate"],
                                scoring="neg_mean_absolute_error", n_repeats=5, random_state=C.RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURES, "importance_MAE_increase_pp": pi.importances_mean * 100,
                        "std_pp": pi.importances_std * 100}).sort_values("importance_MAE_increase_pp", ascending=False)
    imp.round(4).to_csv(C.METRICS_DIR / f"permutation_importance_h{h}.csv", index=False)
    res["permutation_importance"] = imp.round(4).to_dict(orient="records")
    res["permutation_seconds"] = round(time.perf_counter() - t0, 1)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    imp_s = imp.sort_values("importance_MAE_increase_pp")
    ax.barh(imp_s["feature"], imp_s["importance_MAE_increase_pp"], xerr=imp_s["std_pp"], color=S.ACCENT, alpha=0.9)
    ax.set(title=f"Permutation importance on test ({sel_r}, {h} min)", xlabel="Increase in MAE when shuffled (pp)")
    res["figures"]["permutation_importance"] = _save(fig, f"eval_permutation_importance_h{h}.png")
    return res


def unseen_park_experiment(grid: pd.DataFrame, summary: dict, h: int = C.PRIMARY_HORIZON) -> dict:
    """Can the model forecast a car park it never saw during training?

    5-fold GroupKFold over car parks. Each fold trains the selected model type
    (same tuned parameters) on train+val dates of the *other* parks, with the
    car-park ID removed, and is scored on the test period of the held-out parks.
    """
    s = summary[str(h)]
    name = s["selected_regressor"]
    est = regression_candidates()[name][0]
    params = {k: (None if v == "None" else v) for k, v in s["regression"][name]["best_params"].items()}
    feats = build_features(grid, h, codes=park_codes(grid))
    feats["park_code"] = 0  # hide identity: the model must rely on occupancy history + time
    sp = split_by_date(feats)
    tv = pd.concat([sp["train"], sp["val"]])
    te = sp["test"]
    parks = np.array(sorted(feats["SystemCodeNumber"].unique()))
    gkf = GroupKFold(n_splits=5)
    errs = []
    for _, held_idx in gkf.split(parks, groups=parks):
        held = parks[held_idx]
        m = clone(est).set_params(**params).fit(tv.loc[~tv["SystemCodeNumber"].isin(held), FEATURES],
                                                 tv.loc[~tv["SystemCodeNumber"].isin(held), "target_rate"])
        t = te[te["SystemCodeNumber"].isin(held)]
        errs.append(pd.DataFrame({"park": t["SystemCodeNumber"], "err": np.abs(t["target_rate"] - np.clip(m.predict(t[FEATURES]), 0, 1)),
                                  "pers": np.abs(t["target_rate"] - t["rate_now"])}))
    e = pd.concat(errs)
    seen = pd.read_csv(C.PREDICTIONS_DIR / f"test_predictions_h{h}.csv")
    seen_mae = mean_absolute_error(seen["target_rate"], seen[f"reg::{name}"])
    return {"horizon_minutes": h, "model": name, "n_test_rows": int(len(e)),
            "unseen_park_MAE_pp": float(e["err"].mean() * 100),
            "seen_park_MAE_pp": float(seen_mae * 100),
            "persistence_MAE_pp": float(e["pers"].mean() * 100)}


def main():
    C.ensure_dirs()
    summary = json.loads((C.METRICS_DIR / "training_summary.json").read_text())
    grid = load_grid()
    out = {}
    for h in sorted(int(k) for k in summary):
        print(f"Evaluating horizon {h} ...")
        out[str(h)] = evaluate_horizon(h, summary, grid)
        out[str(h)]["selected_regressor"] = summary[str(h)]["selected_regressor"]
        out[str(h)]["selected_classifier"] = summary[str(h)]["selected_classifier"]
    if str(C.PRIMARY_HORIZON) in summary:
        print("Unseen car-park experiment ...")
        out["unseen_park_experiment"] = unseen_park_experiment(grid, summary)
    (C.METRICS_DIR / "evaluation_summary.json").write_text(json.dumps(out, indent=2, default=str))
    for h in [k for k in out if k.isdigit()]:
        print(f"\n--- {h} min ---")
        print(pd.DataFrame(out[h]["regression_table"])[["Model", "Test MAE (pp)", "Test RMSE (pp)", "Test R2",
                                                         "Test MAE (spaces)", "Walk-forward MAE (pp)"]].to_string(index=False))
        print(pd.DataFrame(out[h]["classification_table"])[["Model", "accuracy", "macro_F1", "FULL_recall",
                                                             "FULL_precision", "Walk-forward macro F1"]].to_string(index=False))
        print("interval coverage", round(out[h]["interval_test_coverage_90"], 3), "segments", json.dumps(out[h]["segments"]))
        if "interval_band_report" in out[h]:
            print("band intervals", json.dumps(out[h]["interval_band_report"], indent=1))
    print(json.dumps(out.get("unseen_park_experiment"), indent=2))


if __name__ == "__main__":
    main()
