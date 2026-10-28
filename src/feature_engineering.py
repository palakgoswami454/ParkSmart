"""Turn the 30-minute grid into a supervised forecasting table.

Each row is a *forecast origin*: car park ``p`` at slot ``t`` on some day.
The target is the occupancy rate of the same car park at slot ``t + h`` on the
same day (h = 1 slot for 30 min ahead, h = 2 slots for 60 min ahead).

Leakage rules enforced here:
* Every feature is computed from values observed at or before the origin time
  ``t`` (or from earlier calendar days). Nothing from ``t+1`` onward is used.
* Lags never cross the overnight gap (16:30 -> 08:00 is not "30 minutes ago").
* Calendar features of the target time (its slot, weekday) are allowed because
  the clock time we are forecasting for is known in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

CATEGORICAL_FEATURES = ["park_code"]
NUMERIC_FEATURES = [
    "capacity",
    "origin_slot",
    "target_slot",
    "target_hour",
    "day_of_week",
    "is_weekend",
    "rate_now",
    "rate_lag1",
    "rate_lag2",
    "rate_delta1",
    "rate_roll3_mean",
    "rate_roll3_std",
    "target_slot_rate_1d_ago",
    "target_slot_rate_7d_ago",
    "now_slot_rate_7d_ago",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

FEATURE_DESCRIPTIONS = {
    "park_code": "Integer ID of the car park (categorical).",
    "capacity": "Total spaces in the car park.",
    "origin_slot": "Slot index of the forecast origin (0 = 08:00 ... 17 = 16:30).",
    "target_slot": "Slot index of the time being forecast.",
    "target_hour": "Clock time being forecast, in hours (e.g. 13.5 = 13:30).",
    "day_of_week": "0 = Monday ... 6 = Sunday.",
    "is_weekend": "1 on Saturday/Sunday.",
    "rate_now": "Occupancy rate observed at the origin slot.",
    "rate_lag1": "Occupancy rate 30 minutes before the origin (same day).",
    "rate_lag2": "Occupancy rate 60 minutes before the origin (same day).",
    "rate_delta1": "rate_now - rate_lag1 (short-term trend).",
    "rate_roll3_mean": "Mean of the last three observed slots (origin and two before).",
    "rate_roll3_std": "Std. dev. of the last three observed slots.",
    "target_slot_rate_1d_ago": "Occupancy rate at the target clock time one calendar day earlier.",
    "target_slot_rate_7d_ago": "Occupancy rate at the target clock time on the same weekday last week.",
    "now_slot_rate_7d_ago": "Occupancy rate at the origin clock time on the same weekday last week.",
}


def park_codes(grid: pd.DataFrame) -> dict[str, int]:
    return {p: i for i, p in enumerate(sorted(grid["SystemCodeNumber"].unique()))}


def _shift_days(g: pd.DataFrame, full: pd.DataFrame, days: int, slot_col: str, name: str) -> np.ndarray:
    """Look up the occupancy rate at (date - days, slot_col) for each row of ``g``.

    ``full`` is the complete grid, so every slot of earlier days is available.
    """
    ref = full[["SystemCodeNumber", "date", "slot", "occupancy_rate"]].copy()
    ref["date"] = ref["date"] + pd.Timedelta(days=days)
    ref = ref.rename(columns={"slot": slot_col, "occupancy_rate": name})
    merged = g[["SystemCodeNumber", "date", slot_col]].merge(
        ref, on=["SystemCodeNumber", "date", slot_col], how="left"
    )
    return merged[name].to_numpy()


def build_features(grid: pd.DataFrame, horizon_minutes: int = C.PRIMARY_HORIZON,
                   require_target: bool = True, codes: dict[str, int] | None = None) -> pd.DataFrame:
    """Build the feature table for one forecast horizon.

    Parameters
    ----------
    grid : output of ``data_preprocessing.to_slot_grid`` / ``load_grid``.
    horizon_minutes : 30 or 60.
    require_target : drop rows whose target slot has no observation (training);
        set False for inference at origins whose future is unknown.
    """
    if horizon_minutes not in C.HORIZONS:
        raise ValueError(f"horizon_minutes must be one of {list(C.HORIZONS)}")
    h = C.HORIZONS[horizon_minutes]
    codes = codes or park_codes(grid)

    g = grid.sort_values(["SystemCodeNumber", "date", "slot"]).reset_index(drop=True).copy()
    by_day = g.groupby(["SystemCodeNumber", "date"], sort=False)["occupancy_rate"]

    g["rate_now"] = g["occupancy_rate"]
    g["rate_lag1"] = by_day.shift(1)
    g["rate_lag2"] = by_day.shift(2)
    g["target_rate"] = by_day.shift(-h)  # the label: h slots later, same day
    g["rate_delta1"] = g["rate_now"] - g["rate_lag1"]
    last3 = g[["rate_now", "rate_lag1", "rate_lag2"]]
    g["rate_roll3_mean"] = last3.mean(axis=1, skipna=True)
    g["rate_roll3_std"] = last3.std(axis=1, skipna=True, ddof=0)

    g["origin_slot"] = g["slot"]
    g["target_slot"] = g["slot"] + h
    g = g[g["target_slot"] < C.SLOTS_PER_DAY].copy()  # target must be on the same day
    g["target_time"] = g["slot_time"] + pd.Timedelta(minutes=horizon_minutes)
    g["target_hour"] = 8 + g["target_slot"] * C.SLOT_MINUTES / 60
    g["day_of_week"] = g["date"].dt.dayofweek
    g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
    g["park_code"] = g["SystemCodeNumber"].map(codes).astype(int)

    full = grid
    g["target_slot_rate_1d_ago"] = _shift_days(g, full, 1, "target_slot", "v")
    g["target_slot_rate_7d_ago"] = _shift_days(g, full, 7, "target_slot", "v")
    g["now_slot_rate_7d_ago"] = _shift_days(g, full, 7, "origin_slot", "v")

    # A forecast is only issued when we have a current reading at the origin.
    g = g[g["rate_now"].notna()]
    if require_target:
        g = g[g["target_rate"].notna()]
        g["target_status"] = C.status_from_rate(g["target_rate"])
    g["horizon_minutes"] = horizon_minutes
    keep = ["SystemCodeNumber", "date", "slot_time", "target_time", "horizon_minutes"] + FEATURES
    keep += [c for c in ("target_rate", "target_status") if c in g.columns]
    return g[keep].reset_index(drop=True)


def split_by_date(df: pd.DataFrame, date_col: str = "date") -> dict[str, pd.DataFrame]:
    """Chronological train / validation / test split (no shuffling)."""
    d = pd.to_datetime(df[date_col])
    train = df[d <= C.TRAIN_END]
    val = df[(d > C.TRAIN_END) & (d <= C.VAL_END)]
    test = df[d > C.VAL_END]
    return {"train": train, "val": val, "test": test}


def walk_forward_folds(df: pd.DataFrame, n_folds: int = C.WALK_FORWARD_FOLDS,
                       date_col: str = "date"):
    """Expanding-window folds over the given (train+val) frame.

    The last ``n_folds`` weeks of distinct dates each become one validation
    block; the model for each block is trained on every earlier date only.
    """
    dates = np.array(sorted(pd.to_datetime(df[date_col]).unique()))
    block = 7
    folds = []
    for k in range(n_folds, 0, -1):
        start = len(dates) - k * block
        end = start + block
        val_dates = dates[start:end]
        train_dates = dates[:start]
        tr_idx = df.index[pd.to_datetime(df[date_col]).isin(train_dates)]
        va_idx = df.index[pd.to_datetime(df[date_col]).isin(val_dates)]
        folds.append((tr_idx, va_idx, (str(pd.Timestamp(val_dates[0]).date()), str(pd.Timestamp(val_dates[-1]).date()))))
    return folds
