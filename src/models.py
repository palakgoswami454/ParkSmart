"""Model definitions: baselines, ML candidates and their tuning grids.

All models are scikit-learn compatible estimators that take the feature frame
produced by ``feature_engineering.build_features`` (columns ``FEATURES``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src.feature_engineering import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


# ----------------------------------------------------------------------------
# Baselines (no learning, or trivially simple learning)
# ----------------------------------------------------------------------------
class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """Naive forecast: occupancy in h minutes = occupancy now."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.clip(np.asarray(X["rate_now"], dtype=float), 0, 1)


class HistoricalProfileRegressor(RegressorMixin, BaseEstimator):
    """Average training occupancy for (car park, weekday, target clock time).

    This is what a conventional "typical occupancy" chart shows. It ignores
    what is happening today, so it is a useful second baseline.
    """

    def fit(self, X, y):
        d = pd.DataFrame({"p": X["park_code"].values, "d": X["day_of_week"].values,
                          "s": X["target_slot"].values, "y": np.asarray(y, dtype=float)})
        self.table_ = d.groupby(["p", "d", "s"])["y"].mean()
        self.park_slot_ = d.groupby(["p", "s"])["y"].mean()
        self.global_ = float(d["y"].mean())
        return self

    def predict(self, X):
        key = pd.MultiIndex.from_arrays([X["park_code"].values, X["day_of_week"].values, X["target_slot"].values])
        out = self.table_.reindex(key).to_numpy()
        fb = self.park_slot_.reindex(pd.MultiIndex.from_arrays([X["park_code"].values, X["target_slot"].values])).to_numpy()
        out = np.where(np.isnan(out), fb, out)
        return np.where(np.isnan(out), self.global_, out)


class WeeklyShapePersistenceRegressor(RegressorMixin, BaseEstimator):
    """Strong heuristic baseline: current occupancy + how much it changed over
    the same interval on the same weekday last week.

        forecast = rate_now + (rate_7d_ago[target time] - rate_7d_ago[origin time])

    Falls back to plain persistence when last week's values are missing.
    """

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def predict(self, X):
        now = np.asarray(X["rate_now"], dtype=float)
        change = np.asarray(X["target_slot_rate_7d_ago"], dtype=float) - np.asarray(X["now_slot_rate_7d_ago"], dtype=float)
        return np.clip(np.where(np.isnan(change), now, now + change), 0, 1)


class MajorityClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        vals, counts = np.unique(np.asarray(y), return_counts=True)
        self.classes_ = np.array(C.STATUS_LABELS, dtype=object)
        self.majority_ = vals[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_, dtype=object)


class PersistenceClassifier(ClassifierMixin, BaseEstimator):
    """Status in h minutes = status now."""

    def fit(self, X, y=None):
        self.classes_ = np.array(C.STATUS_LABELS, dtype=object)
        return self

    def predict(self, X):
        return C.status_from_rate(np.asarray(X["rate_now"], dtype=float))


class ThresholdedRegressorClassifier(ClassifierMixin, BaseEstimator):
    """Classify by thresholding a fitted regressor's occupancy-rate forecast."""

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y=None, y_rate=None):
        # The regressor is fitted separately (on the rate target) and passed in.
        self.classes_ = np.array(C.STATUS_LABELS, dtype=object)
        return self

    def predict(self, X):
        return C.status_from_rate(np.clip(self.regressor.predict(X), 0, 1))


# ----------------------------------------------------------------------------
# Preprocessing for models that need it
# ----------------------------------------------------------------------------
def _linear_preprocessor() -> ColumnTransformer:
    """One-hot park ID, impute + scale numeric features (needed for linear models)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("park", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", numeric, NUMERIC_FEATURES),
    ])


def _select_features() -> ColumnTransformer:
    """Trees need no scaling and handle NaN natively; just select columns."""
    return ColumnTransformer([("all", "passthrough", FEATURES)])


CAT_MASK = [f in CATEGORICAL_FEATURES for f in FEATURES]


# ----------------------------------------------------------------------------
# Candidate sets
# ----------------------------------------------------------------------------
def regression_candidates() -> dict:
    """name -> (estimator, param grid for validation tuning, short justification)."""
    return {
        "Persistence (baseline)": (PersistenceRegressor(), {},
            "Naive 'nothing changes' forecast; any useful model must beat it."),
        "Historical profile (baseline)": (HistoricalProfileRegressor(), {},
            "Average occupancy for this car park, weekday and time - what static 'busy times' charts show."),
        "Persistence + weekly shape (baseline)": (WeeklyShapePersistenceRegressor(), {},
            "Current occupancy plus last week's change over the same interval; a strong hand-crafted rule."),
        "Ridge Regression": (
            Pipeline([("prep", _linear_preprocessor()), ("model", Ridge())]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            "Interpretable linear model; tests whether a linear combination of lags and time explains the target."),
        "Random Forest": (
            Pipeline([("prep", _select_features()),
                      ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=C.RANDOM_STATE))]),
            {"model__max_depth": [10, 20, None], "model__min_samples_leaf": [1, 5, 20],
             "model__max_features": [0.5, 1.0]},
            "Non-linear bagged trees; captures interactions (e.g. park x time) and is robust to outliers."),
        "HistGradientBoosting": (
            Pipeline([("prep", _select_features()),
                      ("model", HistGradientBoostingRegressor(categorical_features=CAT_MASK, random_state=C.RANDOM_STATE,
                                                              early_stopping=False))]),
            {"model__learning_rate": [0.03, 0.1], "model__max_iter": [300, 600],
             "model__max_leaf_nodes": [15, 31, 63], "model__l2_regularization": [0.0, 1.0]},
            "Boosted trees; usually strongest on tabular data, native NaN and categorical support, fast."),
    }


def classification_candidates() -> dict:
    return {
        "Majority class (baseline)": (MajorityClassifier(), {}, "Always predicts the most common status."),
        "Persistence (baseline)": (PersistenceClassifier(), {}, "Status in h minutes = status now."),
        "Logistic Regression": (
            Pipeline([("prep", _linear_preprocessor()),
                      ("model", LogisticRegression(max_iter=2000))]),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
            "Linear multinomial classifier; interpretable reference."),
        "Random Forest": (
            Pipeline([("prep", _select_features()),
                      ("model", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=C.RANDOM_STATE))]),
            {"model__max_depth": [10, 20, None], "model__min_samples_leaf": [1, 5],
             "model__class_weight": [None, "balanced"]},
            "Non-linear ensemble; class_weight tested for the rarer FULL/HIGH classes."),
        "HistGradientBoosting": (
            Pipeline([("prep", _select_features()),
                      ("model", HistGradientBoostingClassifier(categorical_features=CAT_MASK,
                                                               random_state=C.RANDOM_STATE, early_stopping=False))]),
            {"model__learning_rate": [0.03, 0.1], "model__max_iter": [300], "model__max_leaf_nodes": [15, 31],
             "model__class_weight": [None, "balanced"]},
            "Boosted trees; strong tabular classifier with class weighting option."),
    }
