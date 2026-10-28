"""Inference: load a saved ParkSmart model and forecast occupancy/availability.

IMPORTANT: the Birmingham data is historical (Oct-Dec 2016). ParkSmart works in
*replay* mode - you pick an "as-of" moment inside the dataset, the model sees
only data up to that moment, and forecasts 30 or 60 minutes ahead. Because the
future is in the dataset, we can show the actual value next to the forecast.
Nothing here is a live sensor reading.

CLI example:
    python -m src.predict --park "Broad Street" --at "2016-12-14 11:00" --horizon 60
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.data_preprocessing import load_grid
from src.feature_engineering import FEATURES, build_features


class ModelNotFoundError(FileNotFoundError):
    pass


def load_bundle(horizon: int = C.PRIMARY_HORIZON) -> dict:
    path = C.MODELS_DIR / f"parksmart_h{horizon}.joblib"
    if not path.exists():
        raise ModelNotFoundError(f"{path} not found. Train first: python -m src.train")
    return joblib.load(path)


def interval_halfwidth(bundle: dict, target_hour) -> np.ndarray:
    """90% range half-width for each forecast: per time-of-day band when the
    bundle has a band table (see src/intervals.py), else the single global width."""
    table = bundle.get("interval_table_90")
    if table:
        from src.intervals import width_for
        return width_for(target_hour, table)
    return np.full(len(target_hour), bundle["interval_halfwidth_90"], dtype=float)


def inference_table(grid: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Features for every origin that has a current reading, with predictions.

    ``actual_rate`` is attached where the future slot exists in the data (for
    replay comparison only - it is never used as an input).
    """
    h = bundle["horizon_minutes"]
    feats = build_features(grid, h, require_target=False, codes=bundle["park_codes"])
    rate = np.clip(bundle["regressor"].predict(feats[FEATURES]), 0, 1)
    feats["pred_rate"] = rate
    feats["pred_status_from_rate"] = C.status_from_rate(rate)
    feats["pred_status"] = bundle["classifier"].predict(feats[FEATURES])
    feats["interval_halfwidth"] = interval_halfwidth(bundle, feats["target_hour"])
    feats["pred_low"] = np.clip(rate - feats["interval_halfwidth"], 0, 1)
    feats["pred_high"] = np.clip(rate + feats["interval_halfwidth"], 0, 1)
    feats["pred_occupied"] = np.round(rate * feats["capacity"]).astype(int)
    feats["pred_available"] = np.maximum(0, feats["capacity"] - feats["pred_occupied"])
    actual = grid[["SystemCodeNumber", "slot_time", "occupancy_rate"]].rename(
        columns={"slot_time": "target_time", "occupancy_rate": "actual_rate"})
    feats = feats.merge(actual, on=["SystemCodeNumber", "target_time"], how="left")
    feats["actual_status"] = pd.Series(pd.NA, index=feats.index, dtype=object)
    has = feats["actual_rate"].notna()
    feats.loc[has, "actual_status"] = C.status_from_rate(feats.loc[has, "actual_rate"])
    return feats


@dataclass
class Forecast:
    park: str
    as_of: str
    forecast_for: str
    horizon_minutes: int
    capacity: int
    current_rate: float
    predicted_rate: float
    interval_90: tuple
    predicted_status: str
    estimated_occupied: int
    estimated_available: int
    actual_rate: float | None
    actual_status: str | None
    model: str

    def as_dict(self):
        return asdict(self)


def forecast(table: pd.DataFrame, bundle: dict, park: str, as_of) -> Forecast:
    as_of = pd.Timestamp(as_of)
    if park not in bundle["park_codes"]:
        raise ValueError(f"Unknown car park '{park}'.")
    row = table[(table["SystemCodeNumber"] == park) & (table["slot_time"] == as_of)]
    if row.empty:
        raise ValueError(
            f"No reading for {park} at {as_of:%Y-%m-%d %H:%M}. Forecasts need a current observation; "
            f"pick another time (08:00-{'16:00' if bundle['horizon_minutes'] == 30 else '15:30'}, 30-min steps)."
        )
    r = row.iloc[0]
    actual = None if pd.isna(r["actual_rate"]) else float(r["actual_rate"])
    return Forecast(
        park=park, as_of=f"{as_of:%Y-%m-%d %H:%M}", forecast_for=f"{r['target_time']:%Y-%m-%d %H:%M}",
        horizon_minutes=int(bundle["horizon_minutes"]), capacity=int(r["capacity"]),
        current_rate=float(r["rate_now"]), predicted_rate=float(r["pred_rate"]),
        interval_90=(float(r["pred_low"]), float(r["pred_high"])),
        predicted_status=str(r["pred_status"]), estimated_occupied=int(r["pred_occupied"]),
        estimated_available=int(r["pred_available"]), actual_rate=actual,
        actual_status=None if actual is None else str(r["actual_status"]),
        model=f"{bundle['regressor_name']} (rate) / {bundle['classifier_name']} (status)",
    )


def main():
    ap = argparse.ArgumentParser(description="ParkSmart replay forecast")
    ap.add_argument("--park", required=True)
    ap.add_argument("--at", required=True, help="as-of time, e.g. '2016-12-14 11:00'")
    ap.add_argument("--horizon", type=int, default=C.PRIMARY_HORIZON, choices=list(C.HORIZONS))
    a = ap.parse_args()
    bundle = load_bundle(a.horizon)
    table = inference_table(load_grid(), bundle)
    f = forecast(table, bundle, a.park, a.at)
    for k, v in f.as_dict().items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
