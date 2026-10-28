"""Transparent parking recommendations for the Find Parking page.

Score (0–1) = 0.6 × proximity + 0.4 × evidence
  proximity = max(0, 1 − straight-line distance / search radius)
  evidence  = weight of the facility's status (see src/data_loader.STATUS):
              reported operational 0.9 · private venue 0.7 · contract ended 0.5 ·
              announced 0.4 · directory listing 0.35 · conflicting reports 0.2

Hard filters (a facility is moved to "not suitable", never silently dropped):
  * status is non-operational or restricted;
  * the sources list vehicle types and the chosen vehicle is not among them;
  * a published tariff exists and the cost for the stay exceeds the budget;
  * it is farther than the search radius.
Unknown information (no tariff, vehicle types not stated) never excludes a
facility; it is reported as missing so the user knows what to check.

No occupancy enters the score: there is no Pune occupancy data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import VEHICLES, PuneData
from src.location_services import add_distances

W_PROXIMITY, W_EVIDENCE = 0.6, 0.4


def tariff_cost(tariff: dict | None, vehicle: str, hours: float):
    """Price for one stay of ``hours`` under a banded tariff.

    Bands are [upper_limit_hours, price]; ``None`` as the limit means "any
    longer stay". Returns (price, band description) or (None, reason).
    """
    if tariff is None:
        return None, "No published tariff"
    bands = tariff["bands"].get(vehicle)
    if not bands:
        return None, f"Tariff does not cover {VEHICLES.get(vehicle, vehicle).lower()}s"
    if hours <= 0:
        raise ValueError("Duration must be positive.")
    prev = 0
    for limit, price in bands:
        if limit is None or hours <= limit:
            desc = f"up to {limit} h" if limit is not None else f"over {prev} h"
            if prev and limit is not None:
                desc = f"{prev}–{limit} h"
            return float(price), desc
        prev = limit
    return None, f"Stays over {prev} h are not covered by the published tariff"


@dataclass
class Recommendation:
    ranked: pd.DataFrame       # suitable facilities, best first
    not_suitable: pd.DataFrame  # with a 'reason' column
    no_location: pd.DataFrame   # suitable on paper but no verified coordinates


def _matches_destination(locality: str, destination: dict) -> bool:
    words = [destination["name"].lower()] + [a.lower() for a in destination.get("aliases", [])]
    loc = (locality or "").lower()
    return any(loc and (loc in w or w in loc) for w in words)


def recommend(data: PuneData, destination: dict, vehicle: str, hours: float,
              budget: float | None = None, radius_km: float = 5.0) -> Recommendation:
    """Rank facilities for a destination (a row of ``data.destinations``).

    Destinations without verified coordinates (e.g. Camp) can only be matched
    by locality name, so they produce no distance ranking.
    """
    dest_lat, dest_lon = destination.get("lat"), destination.get("lon")
    has_point = dest_lat is not None and not pd.isna(dest_lat)
    if vehicle not in VEHICLES:
        raise ValueError(f"Unknown vehicle type '{vehicle}'.")
    if not 0 < hours <= 72:
        raise ValueError("Parking duration must be between 0 and 72 hours.")
    if radius_km <= 0:
        raise ValueError("Search radius must be positive.")
    if budget is not None and budget < 0:
        raise ValueError("Budget cannot be negative.")

    if has_point:
        f = add_distances(data.facilities, dest_lat, dest_lon)
    else:
        f = data.facilities.copy()
        f["distance_km"] = np.nan
    costs, cost_notes = [], []
    for r in f.itertuples():
        price, note = tariff_cost(data.tariffs.get(r.tariff_id) if r.tariff_id else None, vehicle, hours)
        costs.append(price)
        cost_notes.append(note)
    f["cost"] = costs
    f["cost_note"] = cost_notes

    reasons = []
    for r in f.itertuples():
        why = []
        if not r.recommendable:
            why.append(r.status_label)
        if r.vehicle_types and vehicle not in r.vehicle_types:
            why.append(f"No {VEHICLES[vehicle].lower()} parking listed")
        if budget is not None and r.cost is not None and not np.isnan(r.cost) and r.cost > budget:
            why.append(f"₹{r.cost:.0f} is over your ₹{budget:.0f} budget")
        if pd.notna(r.distance_km) and r.distance_km > radius_km:
            why.append(f"{r.distance_km:.1f} km away (beyond {radius_km:g} km)")
        reasons.append("; ".join(why))
    f["reason"] = reasons

    def missing_info(r):
        gaps = ["live occupancy"]
        if not r["vehicle_types"]:
            gaps.append("vehicle types")
        if r["cost"] is None or (isinstance(r["cost"], float) and np.isnan(r["cost"])):
            gaps.append("price")
        if r["capacity"] is None or (isinstance(r["capacity"], float) and np.isnan(r["capacity"])):
            gaps.append("capacity")
        if not r["hours"]:
            gaps.append("opening hours")
        return ", ".join(gaps)

    f["missing"] = f.apply(missing_info, axis=1)
    ok = f[f["reason"] == ""]
    located = ok[ok["has_location"]].copy()
    located["proximity"] = (1 - located["distance_km"] / radius_km).clip(lower=0)
    located["score"] = W_PROXIMITY * located["proximity"] + W_EVIDENCE * located["evidence"]
    located = located.sort_values(["score", "distance_km"], ascending=[False, True])
    no_loc = ok[~ok["has_location"] & ok["locality"].map(lambda l: _matches_destination(l, destination))].copy()
    if not has_point:
        located = located.iloc[0:0]
    bad = f[(f["reason"] != "") & (f["distance_km"].notna() | f["locality"].map(
        lambda l: _matches_destination(l, destination)))].sort_values("distance_km")
    return Recommendation(ranked=located.reset_index(drop=True), not_suitable=bad.reset_index(drop=True),
                          no_location=no_loc.reset_index(drop=True))
