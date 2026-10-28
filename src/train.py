"""Train, tune and select ParkSmart models.

Protocol (identical for every model, both horizons):
1. Tune hyper-parameters: fit on TRAIN, score on VALIDATION (grid search).
2. Walk-forward CV: with the chosen parameters, 4 expanding-window weekly folds
   over TRAIN+VALIDATION. The final model is SELECTED on this CV score.
3. Refit the chosen parameters on TRAIN+VALIDATION and predict the untouched
   TEST period (last two weeks). Test scores are reported, never used to choose.

Run:  python -m src.train            (both horizons)
      python -m src.train --horizon 60
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid

from src import config as C
from src.data_preprocessing import load_grid
from src.feature_engineering import (FEATURE_DESCRIPTIONS, FEATURES, build_features, park_codes,
                                     split_by_date, walk_forward_folds)
from src.models import ThresholdedRegressorClassifier, classification_candidates, regression_candidates

warnings.filterwarnings("ignore", category=UserWarning)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def _reg_score(y, p):
    return float(mean_absolute_error(y, np.clip(p, 0, 1)))


def _clf_score(y, p):
    return float(f1_score(y, p, average="macro", labels=C.STATUS_LABELS, zero_division=0))


def tune(est, grid, X_tr, y_tr, X_va, y_va, kind):
    """Grid search on the validation block. Returns best params and the log."""
    log = []
    best, best_params = None, {}
    for params in ParameterGrid(grid) if grid else [{}]:
        m = clone(est).set_params(**params)
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        p = m.predict(X_va)
        score = _reg_score(y_va, p) if kind == "reg" else _clf_score(y_va, p)
        log.append({"params": params, "score": round(score, 5), "fit_seconds": round(time.perf_counter() - t0, 2)})
        better = best is None or (score < best if kind == "reg" else score > best)
        if better:
            best, best_params = score, params
    return best_params, best, log


def walk_forward(est, X, y, folds, kind):
    rows, oof = [], pd.Series(np.nan, index=X.index, dtype=object if kind == "clf" else float)
    for tr, va, (d0, d1) in folds:
        m = clone(est).fit(X.loc[tr], y.loc[tr])
        p = m.predict(X.loc[va])
        if kind == "reg":
            p = np.clip(p, 0, 1)
            rows.append({"fold": f"{d0}..{d1}", "MAE": _reg_score(y.loc[va], p), "RMSE": _rmse(y.loc[va], p)})
        else:
            rows.append({"fold": f"{d0}..{d1}", "macro_F1": _clf_score(y.loc[va], p)})
        oof.loc[va] = p
    return rows, oof


def latency(model, X, repeats=3):
    """Median time to predict one batch, reported per 1,000 rows."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) / len(X) * 1000 * 1000)  # ms per 1000 rows


def run_horizon(grid: pd.DataFrame, horizon: int) -> dict:
    print(f"\n================ Horizon: {horizon} minutes ================")
    codes = park_codes(grid)
    feats = build_features(grid, horizon, codes=codes)
    sp = split_by_date(feats)
    tv = pd.concat([sp["train"], sp["val"]])
    folds = walk_forward_folds(tv)
    X_tr, X_va, X_tv, X_te = (d[FEATURES] for d in (sp["train"], sp["val"], tv, sp["test"]))

    out = {"horizon_minutes": horizon,
           "rows": {k: int(len(v)) for k, v in sp.items()},
           "date_ranges": {k: [str(v["date"].min().date()), str(v["date"].max().date())] for k, v in sp.items()},
           "walk_forward_folds": [f[2] for f in folds],
           "regression": {}, "classification": {}}
    test_pred = sp["test"][["SystemCodeNumber", "date", "slot_time", "target_time", "capacity",
                            "rate_now", "target_slot", "day_of_week", "target_rate", "target_status"]].copy()
    fitted_reg, fitted_clf = {}, {}

    # ------------------------- Regression ---------------------------------
    for name, (est, grid_, why) in regression_candidates().items():
        t0 = time.perf_counter()
        params, val_score, log = tune(est, grid_, X_tr, sp["train"]["target_rate"], X_va, sp["val"]["target_rate"], "reg")
        tuned = clone(est).set_params(**params)
        wf_rows, oof = walk_forward(tuned, X_tv, tv["target_rate"], folds, "reg")
        t1 = time.perf_counter()
        final = clone(tuned).fit(X_tv, tv["target_rate"])
        fit_s = time.perf_counter() - t1
        p = np.clip(final.predict(X_te), 0, 1)
        test_pred[f"reg::{name}"] = p
        fitted_reg[name] = (final, oof)
        out["regression"][name] = {
            "why": why, "best_params": {k: (v if v is not None else "None") for k, v in params.items()},
            "validation_MAE": round(val_score, 5), "tuning_log": log,
            "walk_forward": wf_rows,
            "walk_forward_MAE_mean": round(float(np.mean([r["MAE"] for r in wf_rows])), 5),
            "walk_forward_MAE_std": round(float(np.std([r["MAE"] for r in wf_rows])), 5),
            "walk_forward_RMSE_mean": round(float(np.mean([r["RMSE"] for r in wf_rows])), 5),
            "final_fit_seconds": round(fit_s, 2),
            "predict_ms_per_1000_rows": round(latency(final, X_te), 3),
            "total_seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"[reg] {name:32s} val MAE={val_score:.4f}  WF MAE={out['regression'][name]['walk_forward_MAE_mean']:.4f}  params={params}")

    reg_sel = min(out["regression"], key=lambda n: out["regression"][n]["walk_forward_MAE_mean"])
    out["selected_regressor"] = reg_sel
    sel_model, sel_oof = fitted_reg[reg_sel]

    # Split-conformal style interval: 90th percentile of absolute out-of-fold
    # walk-forward residuals (each from a model trained only on earlier data).
    resid = (tv["target_rate"] - sel_oof).abs().dropna()
    q90 = float(np.quantile(resid, 0.90))
    q80 = float(np.quantile(resid, 0.80))
    out["interval"] = {"method": "absolute walk-forward residual quantile (split-conformal style)",
                       "n_residuals": int(len(resid)), "halfwidth_90": round(q90, 5), "halfwidth_80": round(q80, 5)}

    # ----------------------- Classification -------------------------------
    for name, (est, grid_, why) in classification_candidates().items():
        t0 = time.perf_counter()
        params, val_score, log = tune(est, grid_, X_tr, sp["train"]["target_status"], X_va, sp["val"]["target_status"], "clf")
        tuned = clone(est).set_params(**params)
        wf_rows, _ = walk_forward(tuned, X_tv, tv["target_status"], folds, "clf")
        t1 = time.perf_counter()
        final = clone(tuned).fit(X_tv, tv["target_status"])
        fit_s = time.perf_counter() - t1
        test_pred[f"clf::{name}"] = final.predict(X_te)
        fitted_clf[name] = final
        out["classification"][name] = {
            "why": why, "best_params": {k: (v if v is not None else "None") for k, v in params.items()},
            "validation_macro_F1": round(val_score, 5), "tuning_log": log, "walk_forward": wf_rows,
            "walk_forward_macro_F1_mean": round(float(np.mean([r["macro_F1"] for r in wf_rows])), 5),
            "walk_forward_macro_F1_std": round(float(np.std([r["macro_F1"] for r in wf_rows])), 5),
            "final_fit_seconds": round(fit_s, 2),
            "predict_ms_per_1000_rows": round(latency(final, X_te), 3),
            "total_seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"[clf] {name:32s} val F1={val_score:.4f}  WF F1={out['classification'][name]['walk_forward_macro_F1_mean']:.4f}  params={params}")

    # Status derived from the selected regressor (uses its walk-forward OOF predictions).
    name = f"Thresholded {reg_sel}"
    oof_status = C.status_from_rate(sel_oof.dropna())
    wf_rows = []
    for _, va, (d0, d1) in folds:
        wf_rows.append({"fold": f"{d0}..{d1}", "macro_F1": _clf_score(tv.loc[va, "target_status"], oof_status.loc[va])})
    thr = ThresholdedRegressorClassifier(sel_model).fit(X_tv)
    test_pred[f"clf::{name}"] = thr.predict(X_te)
    fitted_clf[name] = thr
    out["classification"][name] = {
        "why": "Status obtained by applying the LOW/MODERATE/HIGH/FULL thresholds to the selected regressor's forecast.",
        "best_params": {}, "validation_macro_F1": None, "tuning_log": [], "walk_forward": wf_rows,
        "walk_forward_macro_F1_mean": round(float(np.mean([r["macro_F1"] for r in wf_rows])), 5),
        "walk_forward_macro_F1_std": round(float(np.std([r["macro_F1"] for r in wf_rows])), 5),
        "final_fit_seconds": 0.0, "predict_ms_per_1000_rows": round(latency(thr, X_te), 3), "total_seconds": 0.0,
    }
    clf_sel = max(out["classification"], key=lambda n: out["classification"][n]["walk_forward_macro_F1_mean"])
    out["selected_classifier"] = clf_sel
    print(f"Selected regressor: {reg_sel} | selected classifier: {clf_sel}")

    # ------------------------- Persist ------------------------------------
    test_pred.to_csv(C.PREDICTIONS_DIR / f"test_predictions_h{horizon}.csv", index=False)
    caps = grid.groupby("SystemCodeNumber")["capacity"].first().astype(int).to_dict()
    bundle = {
        "horizon_minutes": horizon,
        "regressor_name": reg_sel, "regressor": sel_model,
        "classifier_name": clf_sel, "classifier": fitted_clf[clf_sel],
        "features": FEATURES, "feature_descriptions": FEATURE_DESCRIPTIONS,
        "park_codes": codes, "capacities": caps,
        "interval_halfwidth_90": q90, "interval_halfwidth_80": q80,
        "status_bins": C.STATUS_BINS, "status_labels": C.STATUS_LABELS,
        "trained_on": out["date_ranges"]["train"][0] + " .. " + out["date_ranges"]["val"][1],
        "sklearn_version": __import__("sklearn").__version__,
    }
    joblib.dump(bundle, C.MODELS_DIR / f"parksmart_h{horizon}.joblib", compress=3)
    # Also keep every fitted regressor for the unseen-period/feature-importance analysis.
    return out


def main(horizons=None):
    C.ensure_dirs()
    grid = load_grid()
    horizons = horizons or list(C.HORIZONS)
    summary = {}
    for h in horizons:
        summary[str(h)] = run_horizon(grid, h)
    path = C.METRICS_DIR / "training_summary.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(summary)
    path.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, choices=list(C.HORIZONS), default=None)
    a = ap.parse_args()
    main([a.horizon] if a.horizon else None)
