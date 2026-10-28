"""Exploratory data analysis: figures + a JSON of the numbers quoted in the report.

Run:  python -m src.eda
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src import plot_style as S
from src.data_preprocessing import load_grid, load_raw
from src.feature_engineering import NUMERIC_FEATURES, build_features

S.apply()


def _save(fig, name):
    path = C.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(C.ROOT))


def run() -> dict:
    C.ensure_dirs()
    raw = load_raw()
    grid = load_grid()
    g = grid.dropna(subset=["occupancy_rate"]).copy()
    g["hour"] = 8 + g["slot"] * 0.5
    g["dow"] = g["date"].dt.dayofweek
    summary: dict = {"figures": {}}

    # 1. Distribution of occupancy rate with status bands
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(g["occupancy_rate"] * 100, bins=50, color=S.ACCENT, alpha=0.85)
    for b, lab in zip([0] + [x * 100 for x in C.STATUS_BINS], C.STATUS_LABELS):
        ax.axvline(b, color=S.STATUS_COLORS[lab], lw=1.5, ls="--")
        ax.text(b + 1, ax.get_ylim()[1] * 0.92, lab, color=S.STATUS_COLORS[lab], fontsize=8, fontweight="bold")
    ax.set(title="Distribution of occupancy rate (30-min grid)", xlabel="Occupancy rate (%)", ylabel="Slots")
    summary["figures"]["occupancy_distribution"] = _save(fig, "eda_01_occupancy_distribution.png")
    summary["occupancy_rate_mean"] = round(float(g["occupancy_rate"].mean()), 4)
    summary["occupancy_rate_median"] = round(float(g["occupancy_rate"].median()), 4)
    summary["status_share"] = {k: round(float(v), 4) for k, v in
                               g["status"].value_counts(normalize=True).reindex(C.STATUS_LABELS).items()}

    # 2. Hour-of-day profile (mean and IQR across car parks)
    park_hour = g.groupby(["SystemCodeNumber", "hour"])["occupancy_rate"].mean().unstack(0) * 100
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.fill_between(park_hour.index, park_hour.quantile(0.25, axis=1), park_hour.quantile(0.75, axis=1),
                    color=S.ACCENT, alpha=0.18, label="Middle 50% of car parks")
    ax.plot(park_hour.index, park_hour.mean(axis=1), color=S.ACCENT, lw=2.2, label="Average car park")
    ax.set(title="Hour-of-day demand profile", xlabel="Time of day (h)", ylabel="Mean occupancy (%)")
    ax.set_xticks(range(8, 17))
    ax.legend(loc="upper left")
    summary["figures"]["hour_profile"] = _save(fig, "eda_02_hour_of_day_profile.png")
    hp = g.groupby("hour")["occupancy_rate"].mean()
    summary["hour_profile_mean"] = {f"{int(h):02d}:{int((h % 1) * 60):02d}": round(float(v), 4) for h, v in hp.items()}
    summary["peak_time"] = f"{int(hp.idxmax()):02d}:{int((hp.idxmax() % 1) * 60):02d}"
    summary["peak_mean_rate"] = round(float(hp.max()), 4)
    summary["lowest_time"] = f"{int(hp.idxmin()):02d}:{int((hp.idxmin() % 1) * 60):02d}"

    # 3. Day of week x hour heatmap
    hm = g.groupby(["dow", "hour"])["occupancy_rate"].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    im = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd", vmin=hm.values.min(), vmax=hm.values.max())
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, len(hm.columns), 2), [f"{int(h)}:00" for h in hm.columns[::2]])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Mean occupancy (%)")
    ax.set_title("Occupancy by weekday and time of day")
    summary["figures"]["weekday_hour_heatmap"] = _save(fig, "eda_03_weekday_hour_heatmap.png")
    dm = g.groupby("dow")["occupancy_rate"].mean()
    summary["weekday_mean"] = {d: round(float(dm[i]), 4) for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])}
    summary["weekday_vs_weekend_mean"] = {
        "weekday": round(float(g[g["dow"] < 5]["occupancy_rate"].mean()), 4),
        "weekend": round(float(g[g["dow"] >= 5]["occupancy_rate"].mean()), 4)}

    # 4. Facility comparison: mean rate and share of FULL slots
    fac = g.groupby("SystemCodeNumber").agg(mean_rate=("occupancy_rate", "mean"),
                                            full_share=("status", lambda s: (s == "FULL").mean()),
                                            capacity=("capacity", "first")).sort_values("mean_rate")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(fac))
    ax.barh(y, fac["mean_rate"] * 100, color=S.ACCENT, alpha=0.85, label="Mean occupancy (%)")
    ax.barh(y, fac["full_share"] * 100, color=S.STATUS_COLORS["FULL"], height=0.4, label="% of slots FULL (>=90%)")
    ax.set_yticks(y, fac.index, fontsize=8)
    ax.set(title="Car-park comparison", xlabel="%")
    ax.legend(loc="lower right")
    summary["figures"]["facility_comparison"] = _save(fig, "eda_04_facility_comparison.png")
    summary["facilities"] = {p: {"mean_rate": round(float(r.mean_rate), 4), "full_share": round(float(r.full_share), 4),
                                 "capacity": int(r.capacity)} for p, r in fac.iterrows()}
    summary["most_often_full"] = fac.sort_values("full_share", ascending=False).head(5).index.tolist()
    summary["least_used"] = fac.head(5).index.tolist()

    # 5. Daily mean occupancy over the study period (trend + split boundaries)
    daily = g.groupby("date")["occupancy_rate"].mean() * 100
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(daily.index, daily.values, color=S.ACCENT, lw=1.8, marker="o", ms=3)
    for d, lab in ((C.TRAIN_END, "train | val"), (C.VAL_END, "val | test")):
        ax.axvline(pd.Timestamp(d) + pd.Timedelta(hours=12), color=S.MUTED, ls="--", lw=1)
        ax.text(pd.Timestamp(d), ax.get_ylim()[1] - 2, lab, fontsize=8, color=S.MUTED, ha="right")
    ax.set(title="Daily mean occupancy over the study period", ylabel="Mean occupancy (%)")
    fig.autofmt_xdate()
    summary["figures"]["daily_trend"] = _save(fig, "eda_05_daily_trend.png")
    split_means = {}
    for name, (a, b) in {"train": ("2016-10-01", C.TRAIN_END), "val": (C.TRAIN_END, C.VAL_END),
                         "test": (C.VAL_END, "2016-12-31")}.items():
        m = (g["date"] > a) & (g["date"] <= b) if name != "train" else (g["date"] <= b)
        split_means[name] = round(float(g.loc[m, "occupancy_rate"].mean()), 4)
    summary["mean_rate_by_split"] = split_means

    # 6. Missing data map: share of slots observed per park per day
    cov = grid.assign(obs=grid["occupancy_rate"].notna()).groupby(["SystemCodeNumber", "date"])["obs"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(cov.values, aspect="auto", cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(cov)), cov.index, fontsize=7)
    ticks = list(range(0, cov.shape[1], 7))
    ax.set_xticks(ticks, [str(cov.columns[i].date()) for i in ticks], rotation=45, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Share of slots observed")
    ax.set_title("Data coverage (white = complete, black = missing)")
    summary["figures"]["coverage_map"] = _save(fig, "eda_06_missing_data_map.png")
    summary["grid_missing_pct"] = round(float(grid["occupancy_rate"].isna().mean() * 100), 3)
    worst = grid.assign(m=grid["occupancy_rate"].isna()).groupby("SystemCodeNumber")["m"].mean().sort_values(ascending=False)
    summary["parks_most_missing"] = {k: round(float(v * 100), 2) for k, v in worst.head(5).items()}

    # 7. Feature-target association (Spearman, 60-min horizon, training period only)
    feats = build_features(grid, 60)
    tr = feats[feats["date"] <= C.TRAIN_END]
    corr = tr[NUMERIC_FEATURES + ["target_rate"]].corr(method="spearman")["target_rate"].drop("target_rate").sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(corr.index, corr.values, color=[S.ACCENT if v > 0 else S.ACCENT2 for v in corr.values])
    ax.set(title="Spearman correlation with occupancy 60 min ahead\n(training period only)", xlabel="rho")
    summary["figures"]["feature_correlation"] = _save(fig, "eda_07_feature_target_correlation.png")
    summary["spearman_with_target_h60"] = {k: round(float(v), 4) for k, v in corr.sort_values(ascending=False).items()}

    # 8. Example daily curves for three contrasting car parks
    picks = [fac.index[-1], fac.index[len(fac) // 2], fac.index[0]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, p in zip(axes, picks):
        sub = g[g["SystemCodeNumber"] == p]
        for _, day in sub.groupby("date"):
            ax.plot(day["hour"], day["occupancy_rate"] * 100, color=S.ACCENT, alpha=0.08, lw=1)
        ax.plot(sub.groupby("hour")["occupancy_rate"].mean().index,
                sub.groupby("hour")["occupancy_rate"].mean() * 100, color=S.INK, lw=2)
        ax.set_title(p, fontsize=10)
        ax.set_xlabel("Time of day (h)")
    axes[0].set_ylabel("Occupancy (%)")
    fig.suptitle("Every day overlaid (thin) vs. average (thick) for three car parks", fontsize=11, fontweight="bold")
    summary["figures"]["daily_curves"] = _save(fig, "eda_08_daily_curves_examples.png")
    summary["example_parks"] = picks

    summary["raw_rows"] = int(len(raw))
    (C.METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("facilities", "figures")}, indent=2))
    return summary


if __name__ == "__main__":
    run()
