"""Bring-your-own-data: test whether the ParkSmart model transfers to a new car park.

The main models use the car-park ID, so they only work for the 28 Birmingham
car parks. This module trains a *park-agnostic* version (ID hidden) and lets a
user upload occupancy counts for any car park — for example a Pune operator's
log. Before showing any forecast, it measures the model's error on the
uploaded data and compares it with persistence ("no change"). Only if the
model beats persistence on the user's own data is the forecast presented as
usable. This is how ParkSmart would be adapted to Pune *with evidence*.

Run ``python -m src.transfer`` to train the park-agnostic models, write the
demo sample (a Birmingham car park the transfer model never saw) and save the
held-out evaluation to ``outputs/metrics/transfer_summary.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src import config as C
from src.data_preprocessing import clean, load_grid, load_raw, to_slot_grid
from src.feature_engineering import FEATURES, build_features
from src.intervals import halfwidths, width_for
from src.models import regression_candidates

HOLDOUT_PARK = "Others-CCCPS8"
SAMPLE_CSV = C.DATA_DIR / "samples" / "sample_unseen_carpark.csv"
SUMMARY_JSON = C.METRICS_DIR / "transfer_summary.json"
COLUMN_ALIASES = {
    "facility": ["facility", "SystemCodeNumber", "car_park", "carpark", "park", "location"],
    "capacity": ["capacity", "Capacity", "total_spaces", "spaces"],
    "occupied": ["occupied", "Occupancy", "occupancy", "occupied_spaces", "count"],
    "timestamp": ["timestamp", "LastUpdated", "datetime", "time", "date_time"],
}
MIN_DAYS = 3
MIN_READINGS = 36  # about two days of half-hourly readings per car park


def model_path(h: int):
    return C.MODELS_DIR / f"parksmart_transfer_h{h}.joblib"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _agnostic_features(grid: pd.DataFrame, h: int, require_target=True) -> pd.DataFrame:
    codes = {p: 0 for p in grid["SystemCodeNumber"].unique()}  # identity hidden
    return build_features(grid, h, require_target=require_target, codes=codes)


def train(h: int, summary: dict) -> dict:
    name = summary[str(h)]["selected_regressor"]
    params = {k: (None if v == "None" else v) for k, v in summary[str(h)]["regression"][name]["best_params"].items()}
    est = clone(regression_candidates()[name][0]).set_params(**params)
    grid = load_grid()
    feats = _agnostic_features(grid, h)
    train_rows = feats[feats["SystemCodeNumber"] != HOLDOUT_PARK]
    hold = feats[feats["SystemCodeNumber"] == HOLDOUT_PARK]

    # Park-level cross-validation for the uncertainty range: every residual
    # comes from a model that never saw that car park.
    oof = pd.Series(np.nan, index=train_rows.index)
    for tr, va in GroupKFold(n_splits=5).split(train_rows, groups=train_rows["SystemCodeNumber"]):
        m = clone(est).fit(train_rows.iloc[tr][FEATURES], train_rows.iloc[tr]["target_rate"])
        oof.iloc[va] = np.clip(m.predict(train_rows.iloc[va][FEATURES]), 0, 1)
    table = halfwidths((train_rows["target_rate"] - oof).abs(), train_rows["target_hour"])

    model = clone(est).fit(train_rows[FEATURES], train_rows["target_rate"])
    bundle = {"horizon_minutes": h, "regressor_name": f"{name} (park-agnostic)", "regressor": model,
              "features": FEATURES, "interval_table_90": table, "holdout_park": HOLDOUT_PARK,
              "trained_parks": int(train_rows["SystemCodeNumber"].nunique()),
              "sklearn_version": __import__("sklearn").__version__}
    joblib.dump(bundle, model_path(h), compress=3)

    p = np.clip(model.predict(hold[FEATURES]), 0, 1)
    hw = width_for(hold["target_hour"], table)
    return {"model": bundle["regressor_name"], "holdout_park": HOLDOUT_PARK, "holdout_rows": int(len(hold)),
            "holdout_MAE_pp": float(np.mean(np.abs(hold["target_rate"] - p)) * 100),
            "holdout_persistence_MAE_pp": float(np.mean(np.abs(hold["target_rate"] - hold["rate_now"])) * 100),
            "holdout_interval_coverage": float(np.mean(np.abs(hold["target_rate"] - p) <= hw)),
            "park_cv_interval_global_pp": table["global"] * 100}


def write_sample() -> None:
    raw = load_raw()
    s = raw[raw["SystemCodeNumber"] == HOLDOUT_PARK].copy()
    s["LastUpdated"] = s["LastUpdated"].dt.strftime("%Y-%m-%d %H:%M:%S")
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(SAMPLE_CSV, index=False)


# ---------------------------------------------------------------------------
# Evaluating an upload
# ---------------------------------------------------------------------------
class UploadError(ValueError):
    pass


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    found = {}
    for target, options in COLUMN_ALIASES.items():
        match = next((c for c in df.columns if c in options or c.strip().lower() in [o.lower() for o in options]), None)
        if match is None:
            raise UploadError(f"Could not find a '{target}' column. Accepted names: {', '.join(options)}.")
        found[target] = match
    out = df[[found[k] for k in COLUMN_ALIASES]].copy()
    out.columns = ["SystemCodeNumber", "Capacity", "Occupancy", "LastUpdated"]
    out["SystemCodeNumber"] = out["SystemCodeNumber"].astype(str).str.strip()
    for col in ("Capacity", "Occupancy"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["LastUpdated"] = pd.to_datetime(out["LastUpdated"], errors="coerce")
    bad = out.isna().any(axis=1)
    if bad.all():
        raise UploadError("No usable rows: check that capacity/occupied are numbers and timestamps are dates.")
    out = out[~bad]
    if (out["Capacity"] <= 0).any():
        raise UploadError("Capacity must be positive for every row.")
    out["Capacity"] = out["Capacity"].astype(int)
    out["Occupancy"] = out["Occupancy"].round().astype(int)
    return out


@dataclass
class TransferResult:
    horizon: int
    n_rows_uploaded: int
    n_rows_used: int
    facilities: list
    days: int
    n_scored: int
    model_mae_pp: float | None
    persistence_mae_pp: float | None
    coverage: float | None
    beats_persistence: bool | None
    per_facility: pd.DataFrame
    latest: pd.DataFrame
    notes: list


def evaluate_upload(df: pd.DataFrame, h: int = C.PRIMARY_HORIZON) -> TransferResult:
    path = model_path(h)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python -m src.transfer` first.")
    bundle = joblib.load(path)
    std = standardise_columns(df)
    notes = []
    cleaned, log = clean(std, min_obs=MIN_READINGS)
    # clean() drops car parks with < MIN_OBSERVATIONS_PER_PARK readings; report it.
    if log["sparse_parks_dropped"]:
        notes.append("Dropped car parks with too few readings: " + ", ".join(log["sparse_parks_dropped"]))
    if cleaned.empty:
        raise UploadError(f"Each car park needs at least {MIN_READINGS} readings.")
    grid, glog = to_slot_grid(cleaned)
    if glog["readings_snapped_to_grid"] == 0:
        raise UploadError("No readings fall between 08:00 and 16:30 within 10 minutes of a half-hour slot.")
    days = int(grid["date"].nunique())
    if days < MIN_DAYS:
        raise UploadError(f"Need at least {MIN_DAYS} days of data; found {days}.")
    if glog["readings_outside_grid_dropped"]:
        notes.append(f"{glog['readings_outside_grid_dropped']} readings outside 08:00–16:30 (the model's training "
                     "window) or far from a half-hour mark were ignored.")

    feats = _agnostic_features(grid, h, require_target=False)
    feats["pred"] = np.clip(bundle["regressor"].predict(feats[FEATURES]), 0, 1)
    feats["hw"] = width_for(feats["target_hour"], bundle["interval_table_90"])
    actual = grid[["SystemCodeNumber", "slot_time", "occupancy_rate"]].rename(
        columns={"slot_time": "target_time", "occupancy_rate": "actual"})
    feats = feats.merge(actual, on=["SystemCodeNumber", "target_time"], how="left")
    scored = feats.dropna(subset=["actual"])

    if len(scored):
        e_m = np.abs(scored["actual"] - scored["pred"])
        e_p = np.abs(scored["actual"] - scored["rate_now"])
        model_mae, pers_mae = float(e_m.mean() * 100), float(e_p.mean() * 100)
        cov = float(np.mean(e_m <= scored["hw"]))
        per = scored.assign(e_m=e_m, e_p=e_p).groupby("SystemCodeNumber").agg(
            rows=("pred", "size"), model_MAE_pp=("e_m", lambda x: x.mean() * 100),
            persistence_MAE_pp=("e_p", lambda x: x.mean() * 100)).reset_index()
    else:
        model_mae = pers_mae = cov = None
        per = pd.DataFrame()
        notes.append("No forecast could be checked against a later reading, so accuracy is unknown.")

    last = feats.sort_values("slot_time").groupby("SystemCodeNumber").tail(1)
    last = last.assign(pred_free=lambda d: np.maximum(0, d["capacity"] - np.round(d["pred"] * d["capacity"])).astype(int))
    return TransferResult(horizon=h, n_rows_uploaded=int(len(df)), n_rows_used=int(len(cleaned)),
                          facilities=sorted(grid["SystemCodeNumber"].unique()), days=days, n_scored=int(len(scored)),
                          model_mae_pp=model_mae, persistence_mae_pp=pers_mae, coverage=cov,
                          beats_persistence=None if model_mae is None else model_mae < pers_mae,
                          per_facility=per, latest=last[["SystemCodeNumber", "slot_time", "target_time", "capacity",
                                                         "rate_now", "pred", "hw", "pred_free", "actual"]],
                          notes=notes)


def main():
    C.ensure_dirs()
    summary = json.loads((C.METRICS_DIR / "training_summary.json").read_text())
    write_sample()
    out = {}
    for h in C.HORIZONS:
        out[str(h)] = train(h, summary)
        print(h, json.dumps(out[str(h)], indent=1))
    # Check the upload path end-to-end on the sample file.
    res = evaluate_upload(pd.read_csv(SAMPLE_CSV), C.PRIMARY_HORIZON)
    out["sample_upload_check"] = {"model_MAE_pp": res.model_mae_pp, "persistence_MAE_pp": res.persistence_mae_pp,
                                  "coverage": res.coverage, "rows_scored": res.n_scored}
    SUMMARY_JSON.write_text(json.dumps(out, indent=2))
    print("sample upload:", out["sample_upload_check"])


if __name__ == "__main__":
    main()
