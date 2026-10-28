"""Central configuration for ParkSmart.

Every path, threshold and split date used anywhere in the project lives here so
that the pipeline, the app, the report and the tests all agree.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
GRID_CSV = PROCESSED_DIR / "parking_30min.csv"
QUALITY_JSON = PROCESSED_DIR / "data_quality_report.json"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# ----------------------------------------------------------------------------
# Dataset facts (verified against the UCI page and the raw file)
# ----------------------------------------------------------------------------
UCI_URL = "https://archive.ics.uci.edu/dataset/482/parking+birmingham"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/482/parking+birmingham.zip"
DATASET_DOI = "10.24432/C51K5Z"
RAW_COLUMNS = ["SystemCodeNumber", "Capacity", "Occupancy", "LastUpdated"]
EXPECTED_RAW_ROWS = 35717

# ----------------------------------------------------------------------------
# Time grid
# ----------------------------------------------------------------------------
# Readings arrive roughly every 27/33 minutes between ~08:00 and 16:30.
# We snap them onto a regular 30-minute grid: 08:00, 08:30, ... , 16:30.
SLOT_MINUTES = 30
FIRST_SLOT = "08:00"
LAST_SLOT = "16:30"
SLOTS_PER_DAY = 18  # 08:00 .. 16:30 inclusive
# A reading is only assigned to a slot if it lies within this many minutes of it.
SNAP_TOLERANCE_MIN = 10

# Car parks with fewer raw observations than this are too sparse to model
# (they would contribute almost no complete days).
MIN_OBSERVATIONS_PER_PARK = 500

# ----------------------------------------------------------------------------
# Prediction task
# ----------------------------------------------------------------------------
HORIZONS = {30: 1, 60: 2}  # minutes ahead -> number of 30-min slots ahead
PRIMARY_HORIZON = 60

# Occupancy status bands (occupancy rate = occupied / capacity).
# LOW < 50%, MODERATE 50-75%, HIGH 75-90%, FULL >= 90%.
# 90% is used for FULL because drivers usually cannot find a free bay well
# before a car park is literally 100% occupied (search time rises sharply).
STATUS_BINS = [0.50, 0.75, 0.90]
STATUS_LABELS = ["LOW", "MODERATE", "HIGH", "FULL"]

# ----------------------------------------------------------------------------
# Chronological split (by calendar date, inclusive)
# ----------------------------------------------------------------------------
TRAIN_END = "2016-11-21"  # train: 2016-10-04 .. 2016-11-21
VAL_END = "2016-12-05"  # validation: 2016-11-22 .. 2016-12-05
# test: 2016-12-06 .. 2016-12-19  (last two weeks, never touched while tuning)

WALK_FORWARD_FOLDS = 4  # expanding-window folds over train+validation

RANDOM_STATE = 42


def status_from_rate(rate):
    """Map occupancy rate(s) in [0, 1] to LOW/MODERATE/HIGH/FULL labels."""
    import numpy as np
    import pandas as pd

    idx = np.digitize(np.asarray(rate, dtype=float), STATUS_BINS, right=False)
    labels = np.array(STATUS_LABELS, dtype=object)[idx]
    if isinstance(rate, pd.Series):
        return pd.Series(labels, index=rate.index, name="status")
    return labels


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
