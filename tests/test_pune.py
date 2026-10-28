"""Tests for the Pune directory, recommendations, uncertainty bands, the
transfer (bring-your-own-data) check and the app pages.

Run:  pytest -q
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_loader import PUNE_DIR, STATUS, load_pune, validate
from src.location_services import directions_url, haversine_km, search_destinations
from src.recommendations import recommend, tariff_cost


@pytest.fixture(scope="module")
def data():
    return load_pune()


def raw(name):
    return json.loads((PUNE_DIR / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------- directory integrity
def test_directory_is_valid_and_fully_sourced(data):
    assert len(data.facilities) >= 15
    for _, f in data.facilities.iterrows():
        assert f["sources"], f["id"]
        assert all(s in data.sources for s in f["sources"])
        assert f["status"] in STATUS


def test_no_occupancy_or_availability_fields_for_pune(data):
    """ParkSmart must not carry any (invented) Pune occupancy/availability numbers."""
    forbidden = {"occupancy", "occupied", "available", "free_spaces", "availability", "occupancy_rate"}
    assert not forbidden & set(raw("facilities.json")["facilities"][0].keys())
    assert not forbidden & set(data.facilities.columns)


def test_no_birmingham_names_in_pune_directory(data):
    birmingham = set(pd.read_csv(C.RAW_CSV)["SystemCodeNumber"].unique())
    text = " ".join(data.facilities["name"]) + " " + " ".join(data.facilities["locality"])
    assert not [b for b in birmingham if b in text]
    assert "Birmingham" not in text


def test_validator_rejects_bad_edits():
    fac, dest, pol = raw("facilities.json"), raw("destinations.json"), raw("policy.json")
    sources = {s["id"]: s for s in raw("sources.json")["sources"]}
    assert validate(fac["facilities"], dest["destinations"], sources, fac["tariffs"], pol) == []

    bad = copy.deepcopy(fac["facilities"])
    bad[0]["sources"] = ["made-up-source"]
    bad[1]["lat"], bad[1]["lon"] = 52.48, -1.89  # Birmingham coordinates
    bad[2]["status"] = "live"
    bad[3]["coord_source"] = None
    problems = validate(bad, dest["destinations"], sources, fac["tariffs"], pol)
    assert any("made-up-source" in p for p in problems)
    assert any("outside the Pune area" in p for p in problems)
    assert any("unknown status" in p for p in problems)
    assert any("coord_source" in p for p in problems)


def test_coordinates_are_plausible(data):
    """Every mapped facility should be within 8 km of some destination landmark,
    which catches swapped lat/lon or typos that still fall inside the bbox."""
    dests = data.destinations.dropna(subset=["lat"])
    for _, f in data.facilities[data.facilities["has_location"]].iterrows():
        nearest = min(haversine_km(f["lat"], f["lon"], d["lat"], d["lon"]) for _, d in dests.iterrows())
        assert nearest < 8, f["id"]


# ------------------------------------------------------------- location services
def test_haversine_known_distance():
    # Pune Junction to Pune Airport, both from Wikipedia: about 7.6 km apart.
    assert haversine_km(18.5289, 73.8743, 18.58222, 73.91972) == pytest.approx(7.6, abs=0.5)
    assert haversine_km(18.5, 73.8, 18.5, 73.8) == 0


def test_directions_url_never_invents_a_pin():
    assert "destination=18.528900,73.874300" in directions_url("x", 18.5289, 73.8743)
    url = directions_url("Cantonment Board pay-and-park, MG Road", None, None, "MG Road, Camp")
    assert "search" in url and "destination=" not in url


def test_destination_search(data):
    assert "laxmi-road" in set(search_destinations(data.destinations, "dagdusheth")["id"])
    assert "hinjawadi" in set(search_destinations(data.destinations, "Hinjewadi")["id"])


# ------------------------------------------------------------- tariffs & recommendations
def test_tariff_bands(data):
    metro = data.tariffs["metro-2024"]
    assert tariff_cost(metro, "two_wheeler", 1.5)[0] == 15
    assert tariff_cost(metro, "two_wheeler", 2)[0] == 15
    assert tariff_cost(metro, "car", 3)[0] == 50
    assert tariff_cost(metro, "car", 10)[0] == 80
    pcb = data.tariffs["pcb-2026"]
    assert tariff_cost(pcb, "car", 4)[0] == 30
    assert tariff_cost(pcb, "car", 8)[0] == 60
    assert tariff_cost(pcb, "car", 30)[0] is None          # beyond a full day: not published
    assert tariff_cost(None, "car", 2)[0] is None          # no tariff: unknown, not zero
    assert tariff_cost(metro, "three_wheeler", 2)[0] is None
    with pytest.raises(ValueError):
        tariff_cost(metro, "car", 0)


def dest(data, i):
    return data.destinations.set_index("id").loc[i].to_dict() | {"id": i}


def test_recommendations_are_ordered_and_filtered(data):
    rec = recommend(data, dest(data, "fc-road"), "car", 3, budget=None, radius_km=5)
    assert not rec.ranked.empty
    assert rec.ranked["score"].is_monotonic_decreasing
    assert (rec.ranked["distance_km"] <= 5).all()
    assert not rec.ranked["status"].isin(["closed", "restricted"]).any()
    # vehicle filter: Kalyani Nagar lists two-wheelers only
    rec_car = recommend(data, dest(data, "kalyani-nagar"), "car", 2)
    assert "metro-kalyaninagar" not in set(rec_car.ranked["id"])
    assert "metro-kalyaninagar" in set(rec_car.not_suitable["id"])
    rec_2w = recommend(data, dest(data, "kalyani-nagar"), "two_wheeler", 2)
    assert "metro-kalyaninagar" in set(rec_2w.ranked["id"])


def test_budget_excludes_only_known_prices(data):
    rec = recommend(data, dest(data, "shivajinagar"), "car", 3, budget=40, radius_km=5)
    metro_ids = {"metro-shivajinagar", "metro-districtcourt"}
    assert metro_ids <= set(rec.not_suitable["id"])  # ₹50 > ₹40
    assert "pmc-sane" in set(rec.ranked["id"])        # price unknown: kept, flagged as missing
    assert "price" in rec.ranked.set_index("id").loc["pmc-sane", "missing"]


def test_locality_only_destination(data):
    rec = recommend(data, dest(data, "camp"), "car", 3)
    assert rec.ranked.empty
    assert set(rec.no_location["id"]) == {"pcb-mgroad", "pcb-fashionstreet"}


def test_every_recommendation_reports_missing_live_occupancy(data):
    rec = recommend(data, dest(data, "airport"), "car", 2)
    assert rec.ranked["missing"].str.contains("live occupancy").all()


@pytest.mark.parametrize("kwargs", [dict(vehicle="bus"), dict(hours=0), dict(hours=100), dict(radius_km=0),
                                    dict(budget=-5)])
def test_invalid_inputs_raise(data, kwargs):
    args = dict(vehicle="car", hours=2, budget=None, radius_km=5) | kwargs
    with pytest.raises(ValueError):
        recommend(data, dest(data, "fc-road"), **args)


# ------------------------------------------------------------- intervals & transfer
models_ready = all((C.MODELS_DIR / f"parksmart_h{h}.joblib").exists() for h in C.HORIZONS)


@pytest.mark.skipif(not models_ready, reason="run python -m src.train first")
def test_band_intervals_cover_about_90_percent_on_test():
    ev = json.loads((C.METRICS_DIR / "evaluation_summary.json").read_text())
    for h in ("30", "60"):
        rep = ev[h]["interval_band_report"]
        assert 0.87 <= rep["band_coverage"] <= 0.93
        spread_band = max(v["band_coverage"] for v in rep["by_band"].values()) - min(
            v["band_coverage"] for v in rep["by_band"].values())
        spread_global = max(v["global_coverage"] for v in rep["by_band"].values()) - min(
            v["global_coverage"] for v in rep["by_band"].values())
        assert spread_band < spread_global  # band widths even out coverage across the day


transfer_ready = (C.MODELS_DIR / "parksmart_transfer_h60.joblib").exists()


@pytest.mark.skipif(not transfer_ready, reason="run python -m src.transfer first")
def test_transfer_on_unseen_sample_beats_persistence():
    from src.transfer import SAMPLE_CSV, evaluate_upload
    res = evaluate_upload(pd.read_csv(SAMPLE_CSV), 60)
    assert res.n_scored > 500
    assert res.beats_persistence is True
    assert res.latest["pred"].between(0, 1).all()


@pytest.mark.skipif(not transfer_ready, reason="run python -m src.transfer first")
def test_transfer_rejects_bad_uploads():
    from src.transfer import UploadError, evaluate_upload
    with pytest.raises(UploadError):
        evaluate_upload(pd.DataFrame({"a": [1], "b": [2]}), 60)
    tiny = pd.DataFrame({"facility": ["X"] * 5, "capacity": [100] * 5, "occupied": [10, 20, 30, 40, 50],
                         "timestamp": pd.date_range("2026-01-05 09:00", periods=5, freq="30min").astype(str)})
    with pytest.raises(UploadError):
        evaluate_upload(tiny, 60)
    neg_cap = tiny.assign(capacity=0)
    with pytest.raises(UploadError):
        evaluate_upload(neg_cap, 60)


# ------------------------------------------------------------- app pages
@pytest.mark.skipif(not models_ready, reason="run python -m src.train first")
@pytest.mark.parametrize("page", ["overview", "explore", "find_parking", "city_insights", "forecast",
                                  "model_performance", "about"])
def test_page_runs_without_errors(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.ROOT / "pages" / f"{page}.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_find_parking_form_end_to_end():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.ROOT / "pages" / "find_parking.py"), default_timeout=120)
    at.query_params["dest"] = "airport"
    at.query_params["vehicle"] = "car"
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Pune Airport multi-level car park" in text
    assert "no occupancy data for Pune" in text
