"""Load, validate and clean the UCI Parking Birmingham data.

Output: a regular 30-minute grid (one row per car park x date x slot) saved to
``data/processed/parking_30min.csv`` plus a JSON data-quality report.

Run:  python -m src.data_preprocessing
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C


class DataValidationError(RuntimeError):
    """Raised when the raw file does not look like the UCI dataset."""


# ----------------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------------
def load_raw(path: Path = C.RAW_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download it from {C.UCI_URL} "
            "(see data/README.md) and place dataset.csv in data/raw/."
        )
    df = pd.read_csv(path)
    validate_raw(df)
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="raise")
    df["SystemCodeNumber"] = df["SystemCodeNumber"].astype(str).str.strip()
    return df


def validate_raw(df: pd.DataFrame) -> None:
    missing = [c for c in C.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing expected columns: {missing}")
    if len(df) == 0:
        raise DataValidationError("Raw dataset is empty.")
    for col in ("Capacity", "Occupancy"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column {col} must be numeric.")
    if (df["Capacity"] <= 0).any():
        raise DataValidationError("Found non-positive capacities.")


# ----------------------------------------------------------------------------
# Quality audit (computed on the raw data, before any cleaning)
# ----------------------------------------------------------------------------
def quality_report(df: pd.DataFrame) -> dict:
    rate = df["Occupancy"] / df["Capacity"]
    per_park = df.groupby("SystemCodeNumber").agg(
        observations=("Occupancy", "size"),
        capacity=("Capacity", "first"),
        capacity_values=("Capacity", "nunique"),
        first_reading=("LastUpdated", "min"),
        last_reading=("LastUpdated", "max"),
    )
    per_park["mean_occupancy_rate"] = rate.groupby(df["SystemCodeNumber"]).mean()
    all_days = pd.date_range(df["LastUpdated"].min().normalize(), df["LastUpdated"].max().normalize())
    observed_days = set(df["LastUpdated"].dt.normalize())
    gaps = df.sort_values(["SystemCodeNumber", "LastUpdated"]).groupby("SystemCodeNumber")[
        "LastUpdated"
    ].diff().dt.total_seconds() / 60
    within_day_gaps = gaps[gaps < 240]
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "car_parks": int(df["SystemCodeNumber"].nunique()),
        "first_timestamp": str(df["LastUpdated"].min()),
        "last_timestamp": str(df["LastUpdated"].max()),
        "calendar_days_in_range": int(len(all_days)),
        "days_with_data": int(len(observed_days)),
        "days_without_any_data": [str(d.date()) for d in all_days if d not in observed_days],
        "missing_values_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "missing_value_pct_per_column": {c: round(float(v) * 100, 3) for c, v in df.isna().mean().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_park_timestamp_pairs": int(df.duplicated(["SystemCodeNumber", "LastUpdated"]).sum()),
        "negative_occupancy_rows": int((df["Occupancy"] < 0).sum()),
        "occupancy_above_capacity_rows": int((df["Occupancy"] > df["Capacity"]).sum()),
        "max_occupancy_rate": round(float(rate.max()), 4),
        "min_occupancy_rate": round(float(rate.min()), 4),
        "parks_with_changing_capacity": int((per_park["capacity_values"] > 1).sum()),
        "median_minutes_between_readings": round(float(within_day_gaps.median()), 2),
        "typical_reading_hours": f"{df['LastUpdated'].dt.time.min()} - {df['LastUpdated'].dt.time.max()}",
        "per_park": {
            k: {
                "observations": int(v["observations"]),
                "capacity": int(v["capacity"]),
                "first_reading": str(v["first_reading"]),
                "last_reading": str(v["last_reading"]),
                "mean_occupancy_rate": round(float(v["mean_occupancy_rate"]), 4),
            }
            for k, v in per_park.iterrows()
        },
    }


# ----------------------------------------------------------------------------
# Cleaning and regularisation onto the 30-minute grid
# ----------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicates, fix impossible values and drop sparse car parks."""
    log: dict = {"rows_in": int(len(df))}
    df = df.drop_duplicates().copy()
    log["after_dropping_exact_duplicates"] = int(len(df))

    # Negative occupancy is physically impossible -> sensor/count error -> clip to 0.
    log["negative_occupancy_clipped_to_zero"] = int((df["Occupancy"] < 0).sum())
    df["Occupancy"] = df["Occupancy"].clip(lower=0)

    # Raw occupancy rate. Values slightly above 100% (max ~104%) exist because
    # counts include cars circulating/queuing; the *rate* used for modelling is
    # capped at 1.0 but the raw count is kept for reference.
    df["occupancy_rate_raw"] = df["Occupancy"] / df["Capacity"]
    log["rates_above_1_capped"] = int((df["occupancy_rate_raw"] > 1).sum())
    df["occupancy_rate"] = df["occupancy_rate_raw"].clip(0, 1)

    counts = df["SystemCodeNumber"].value_counts()
    sparse = sorted(counts[counts < C.MIN_OBSERVATIONS_PER_PARK].index.tolist())
    log["sparse_parks_dropped"] = {p: int(counts[p]) for p in sparse}
    df = df[~df["SystemCodeNumber"].isin(sparse)]
    log["rows_out"] = int(len(df))
    log["parks_out"] = int(df["SystemCodeNumber"].nunique())
    return df, log


def to_slot_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Snap readings to the nearest 30-minute slot and build a complete grid.

    Slots with no reading stay NaN (we never invent occupancy values); they are
    simply unavailable as targets and appear as missing lag features.
    """
    log: dict = {}
    d = df.copy()
    d["slot_time"] = d["LastUpdated"].dt.round(f"{C.SLOT_MINUTES}min")
    offset = (d["LastUpdated"] - d["slot_time"]).abs().dt.total_seconds() / 60
    d = d[offset <= C.SNAP_TOLERANCE_MIN]
    tod = d["slot_time"].dt.strftime("%H:%M")
    d = d[(tod >= C.FIRST_SLOT) & (tod <= C.LAST_SLOT)]
    log["readings_snapped_to_grid"] = int(len(d))
    log["readings_outside_grid_dropped"] = int(len(df) - len(d))

    agg = (
        d.groupby(["SystemCodeNumber", "slot_time"])
        .agg(
            capacity=("Capacity", "first"),
            occupied=("Occupancy", "mean"),
            occupancy_rate=("occupancy_rate", "mean"),
            n_readings=("Occupancy", "size"),
        )
        .reset_index()
    )
    log["slots_with_multiple_readings_averaged"] = int((agg["n_readings"] > 1).sum())

    # Complete grid: every park x every day with data x every slot.
    days = pd.Series(sorted(d["slot_time"].dt.normalize().unique()))
    slot_offsets = pd.timedelta_range(
        start=pd.Timedelta(C.FIRST_SLOT + ":00"), periods=C.SLOTS_PER_DAY, freq=f"{C.SLOT_MINUTES}min"
    )
    parks = sorted(d["SystemCodeNumber"].unique())
    full_index = pd.MultiIndex.from_product(
        [parks, [day + off for day in days for off in slot_offsets]],
        names=["SystemCodeNumber", "slot_time"],
    )
    grid = agg.set_index(["SystemCodeNumber", "slot_time"]).reindex(full_index).reset_index()
    caps = d.groupby("SystemCodeNumber")["Capacity"].first()
    grid["capacity"] = grid["SystemCodeNumber"].map(caps).astype(int)
    grid["n_readings"] = grid["n_readings"].fillna(0).astype(int)
    grid["date"] = grid["slot_time"].dt.normalize()
    grid["slot"] = (
        (grid["slot_time"] - grid["date"] - pd.Timedelta(C.FIRST_SLOT + ":00")).dt.total_seconds()
        // (C.SLOT_MINUTES * 60)
    ).astype(int)
    grid["status"] = pd.Series(pd.NA, index=grid.index, dtype=object)
    has = grid["occupancy_rate"].notna()
    grid.loc[has, "status"] = C.status_from_rate(grid.loc[has, "occupancy_rate"])

    log["grid_rows"] = int(len(grid))
    log["grid_missing_slots"] = int(grid["occupancy_rate"].isna().sum())
    log["grid_missing_pct"] = round(float(grid["occupancy_rate"].isna().mean() * 100), 3)
    log["grid_days"] = int(len(days))
    return grid.sort_values(["SystemCodeNumber", "slot_time"]).reset_index(drop=True), log


def run(raw_path: Path = C.RAW_CSV) -> pd.DataFrame:
    C.ensure_dirs()
    raw = load_raw(raw_path)
    report = {"raw_audit": quality_report(raw)}
    cleaned, clean_log = clean(raw)
    grid, grid_log = to_slot_grid(cleaned)
    report["cleaning"] = clean_log
    report["grid"] = grid_log
    report["status_distribution_on_grid"] = {
        k: int(v) for k, v in grid["status"].value_counts().reindex(C.STATUS_LABELS).fillna(0).items()
    }
    grid.to_csv(C.GRID_CSV, index=False)
    C.QUALITY_JSON.write_text(json.dumps(report, indent=2))
    print(f"Saved grid: {C.GRID_CSV}  ({len(grid):,} rows)")
    print(f"Saved quality report: {C.QUALITY_JSON}")
    print(json.dumps({k: report[k] for k in ("cleaning", "grid", "status_distribution_on_grid")}, indent=2))
    return grid


def load_grid(path: Path = C.GRID_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data_preprocessing` first.")
    g = pd.read_csv(path, parse_dates=["slot_time", "date"])
    return g


if __name__ == "__main__":
    run()
