"""Uncertainty ranges for occupancy forecasts.

Two split-conformal-style ranges are built from *walk-forward out-of-fold*
residuals (each residual comes from a model trained only on earlier weeks):

* global   – one half-width = 90th percentile of |residual| over all rows;
* by band  – one half-width per time-of-day band of the forecast target
             ("Mondrian" conformal). The global range under-covered morning
             forecasts (≈87%) and over-covered afternoon ones (≈93%) on test,
             so a band-specific width is the statistically better choice.

Run ``python -m src.intervals`` to (re)compute band widths for the saved
models without re-running the full hyper-parameter search; ``src.train`` calls
the same functions, so a full retrain reproduces exactly the same numbers.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src import config as C

# Target-time bands (hours of day, inclusive lower edge, exclusive upper edge).
BANDS = [("08:30–10:30", 8.0, 10.75), ("11:00–12:30", 10.75, 12.75),
         ("13:00–14:30", 12.75, 14.75), ("15:00–16:30", 14.75, 17.0)]


def band_of(target_hour) -> np.ndarray:
    h = np.asarray(target_hour, dtype=float)
    out = np.empty(len(h), dtype=object)
    for name, lo, hi in BANDS:
        out[(h >= lo) & (h < hi)] = name
    return out


def halfwidths(abs_resid: pd.Series, target_hour: pd.Series, level: float = 0.90) -> dict:
    """Global and per-band half-widths, using the finite-sample conformal quantile."""
    r = pd.Series(np.asarray(abs_resid, dtype=float), index=abs_resid.index)
    ok = r.notna()
    r, th = r[ok], np.asarray(target_hour, dtype=float)[ok.to_numpy()]

    def q(x):
        n = len(x)
        k = min(1.0, np.ceil((n + 1) * level) / n)
        return float(np.quantile(x, k, method="higher"))

    bands = band_of(th)
    by_band = {name: {"halfwidth": q(r[bands == name]), "n": int((bands == name).sum())} for name, _, _ in BANDS}
    return {"level": level, "global": q(r), "n": int(len(r)), "by_band": by_band}


def width_for(target_hour, table: dict) -> np.ndarray:
    names = band_of(target_hour)
    return np.array([table["by_band"][n]["halfwidth"] for n in names], dtype=float)


def coverage_report(y, pred, target_hour, table: dict) -> dict:
    """Empirical coverage of the global and band ranges on a set of predictions."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    err = np.abs(y - pred)
    band_w = width_for(target_hour, table)
    names = band_of(target_hour)
    rows = {}
    for name, _, _ in BANDS:
        m = names == name
        rows[name] = {"n": int(m.sum()),
                      "global_coverage": float(np.mean(err[m] <= table["global"])),
                      "band_coverage": float(np.mean(err[m] <= band_w[m])),
                      "band_halfwidth_pp": table["by_band"][name]["halfwidth"] * 100}
    return {"global_coverage": float(np.mean(err <= table["global"])),
            "band_coverage": float(np.mean(err <= band_w)),
            "global_halfwidth_pp": table["global"] * 100,
            "mean_band_halfwidth_pp": float(np.mean(band_w) * 100),
            "by_band": rows}


def recompute_for_saved_models() -> dict:
    """Re-run walk-forward for each saved model's selected regressor and store band widths."""
    from src.data_preprocessing import load_grid
    from src.feature_engineering import FEATURES, build_features, split_by_date, walk_forward_folds
    from src.models import regression_candidates

    summary_path = C.METRICS_DIR / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    grid = load_grid()
    out = {}
    for h in C.HORIZONS:
        path = C.MODELS_DIR / f"parksmart_h{h}.joblib"
        bundle = joblib.load(path)
        s = summary[str(h)]
        name = s["selected_regressor"]
        params = {k: (None if v == "None" else v) for k, v in s["regression"][name]["best_params"].items()}
        est = clone(regression_candidates()[name][0]).set_params(**params)
        feats = build_features(grid, h, codes=bundle["park_codes"])
        sp = split_by_date(feats)
        tv = pd.concat([sp["train"], sp["val"]])
        oof = pd.Series(np.nan, index=tv.index)
        for tr, va, _ in walk_forward_folds(tv):
            m = clone(est).fit(tv.loc[tr, FEATURES], tv.loc[tr, "target_rate"])
            oof.loc[va] = np.clip(m.predict(tv.loc[va, FEATURES]), 0, 1)
        table = halfwidths((tv["target_rate"] - oof).abs(), tv["target_hour"])
        bundle["interval_table_90"] = table
        joblib.dump(bundle, path, compress=3)
        s["interval"]["by_band_90"] = {k: round(v["halfwidth"], 5) for k, v in table["by_band"].items()}
        s["interval"]["conformal_global_90"] = round(table["global"], 5)
        out[h] = table
        print(f"h={h}: global ±{table['global']*100:.2f} pp; " +
              ", ".join(f"{k} ±{v['halfwidth']*100:.2f}" for k, v in table["by_band"].items()))
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return out


if __name__ == "__main__":
    recompute_for_saved_models()
