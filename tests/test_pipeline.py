"""Verification tests for ParkSmart.

Run:  pytest -q
Data tests need data/raw/dataset.csv; model tests need `python -m src.train`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_preprocessing import clean, load_raw, to_slot_grid
from src.feature_engineering import FEATURES, build_features, split_by_date, walk_forward_folds


@pytest.fixture(scope="session")
def raw():
    return load_raw()


@pytest.fixture(scope="session")
def grid(raw):
    cleaned, _ = clean(raw)
    g, _ = to_slot_grid(cleaned)
    return g


# --------------------------------------------------------------------------- data
def test_raw_matches_uci_description(raw):
    assert len(raw) == C.EXPECTED_RAW_ROWS          # UCI page: 35,717 instances
    assert list(raw.columns) == C.RAW_COLUMNS       # UCI page: 4 features
    assert raw["SystemCodeNumber"].nunique() == 30
    assert raw["LastUpdated"].min() >= pd.Timestamp("2016-10-04")
    assert raw["LastUpdated"].max() < pd.Timestamp("2016-12-20")
    assert raw.isna().sum().sum() == 0


def test_capacity_constant_per_park(raw):
    assert (raw.groupby("SystemCodeNumber")["Capacity"].nunique() == 1).all()


def test_grid_is_regular_and_valid(grid):
    assert not grid.duplicated(["SystemCodeNumber", "slot_time"]).any()
    assert grid["slot"].between(0, C.SLOTS_PER_DAY - 1).all()
    r = grid["occupancy_rate"].dropna()
    assert r.between(0, 1).all()
    # every park has exactly SLOTS_PER_DAY slots on every day of the grid
    assert (grid.groupby(["SystemCodeNumber", "date"]).size() == C.SLOTS_PER_DAY).all()


def test_status_thresholds():
    got = C.status_from_rate(np.array([0.0, 0.499, 0.5, 0.749, 0.75, 0.899, 0.9, 1.0]))
    assert list(got) == ["LOW", "LOW", "MODERATE", "MODERATE", "HIGH", "HIGH", "FULL", "FULL"]


# ------------------------------------------------------------------ leakage
@pytest.mark.parametrize("h", [30, 60])
def test_features_match_grid_and_target_is_future(grid, h):
    f = build_features(grid, h)
    idx = grid.set_index(["SystemCodeNumber", "slot_time"])["occupancy_rate"]
    sample = f.sample(300, random_state=0)
    for r in sample.itertuples():
        assert r.rate_now == pytest.approx(idx[(r.SystemCodeNumber, r.slot_time)])
        assert r.target_rate == pytest.approx(idx[(r.SystemCodeNumber, r.target_time)])
        assert r.target_time - r.slot_time == pd.Timedelta(minutes=h)
        if not np.isnan(r.rate_lag1):
            assert r.rate_lag1 == pytest.approx(idx[(r.SystemCodeNumber, r.slot_time - pd.Timedelta(minutes=30))])
    # forecasts never cross the overnight gap
    assert (f["target_time"].dt.date == f["slot_time"].dt.date).all()


def test_features_do_not_change_when_future_changes(grid):
    """The strongest leakage check: corrupt every value after a cut-off and
    confirm features of origins at or before the cut-off are unchanged."""
    cut = pd.Timestamp("2016-11-15 12:00")
    corrupted = grid.copy()
    corrupted.loc[corrupted["slot_time"] > cut, "occupancy_rate"] = 0.123
    a = build_features(grid, 60, require_target=False)
    b = build_features(corrupted, 60, require_target=False)
    a, b = a[a["slot_time"] <= cut].reset_index(drop=True), b[b["slot_time"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])


def test_chronological_split_has_no_overlap(grid):
    sp = split_by_date(build_features(grid, 60))
    assert sp["train"]["date"].max() < sp["val"]["date"].min()
    assert sp["val"]["date"].max() < sp["test"]["date"].min()
    tv = pd.concat([sp["train"], sp["val"]])
    for tr, va, _ in walk_forward_folds(tv):
        assert tv.loc[tr, "date"].max() < tv.loc[va, "date"].min()


# --------------------------------------------------------------------- model
models_exist = all((C.MODELS_DIR / f"parksmart_h{h}.joblib").exists() for h in C.HORIZONS)


@pytest.mark.skipif(not models_exist, reason="run python -m src.train first")
@pytest.mark.parametrize("h", [30, 60])
def test_saved_model_inference(grid, h):
    from src.predict import forecast, inference_table, load_bundle
    bundle = load_bundle(h)
    t = inference_table(grid, bundle)
    assert t["pred_rate"].between(0, 1).all()
    assert (t["pred_available"] >= 0).all()
    assert (t["pred_available"] == np.maximum(0, t["capacity"] - np.round(t["pred_rate"] * t["capacity"]))).all()
    assert (t["pred_low"] <= t["pred_rate"]).all() and (t["pred_rate"] <= t["pred_high"]).all()
    f = forecast(t, bundle, "Broad Street", "2016-12-14 12:00")
    assert f.horizon_minutes == h and 0 <= f.estimated_available <= f.capacity
    with pytest.raises(ValueError):
        forecast(t, bundle, "Not a car park", "2016-12-14 12:00")


@pytest.mark.skipif(not models_exist, reason="run python -m src.train first")
def test_reported_test_metrics_match_saved_model(grid):
    """Numbers in outputs/metrics must be reproducible from the saved model."""
    from sklearn.metrics import mean_absolute_error
    from src.predict import load_bundle
    ev = json.loads((C.METRICS_DIR / "evaluation_summary.json").read_text())
    for h in C.HORIZONS:
        bundle = load_bundle(h)
        test = split_by_date(build_features(grid, h, codes=bundle["park_codes"]))["test"]
        mae = mean_absolute_error(test["target_rate"], np.clip(bundle["regressor"].predict(test[FEATURES]), 0, 1)) * 100
        reported = [r for r in ev[str(h)]["regression_table"] if r["Selected"]][0]["Test MAE (pp)"]
        assert mae == pytest.approx(reported, abs=1e-3)
