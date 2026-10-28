"""Load and validate the Pune facility directory, destinations, policy notes and sources.

The Pune files live in ``data/pune/`` and are hand-curated from the sources in
``sources.json``. This module refuses to load them if any fact lacks a source,
if a coordinate falls outside the Pune area, or if a status/tariff is unknown,
so a bad edit fails loudly instead of showing unsourced data in the app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src import config as C

PUNE_DIR = C.DATA_DIR / "pune"

# Pune metropolitan bounding box used as a sanity check on every coordinate.
PUNE_BBOX = {"lat": (18.40, 18.70), "lon": (73.65, 74.05)}

VEHICLES = {"cycle": "Cycle", "two_wheeler": "Two-wheeler", "three_wheeler": "Three-wheeler", "car": "Car"}

FACILITY_TYPES = {
    "metro": "Metro station", "railway": "Railway station", "airport": "Airport", "bus": "Bus station",
    "municipal": "Municipal lot", "cantonment": "Cantonment Board lot", "mall": "Mall (private)",
}

# status code -> (label, badge colour, evidence weight used in ranking, may be recommended)
STATUS = {
    "reported":    ("Reported operational", "green", 0.90, True),
    "private":     ("Private venue parking", "blue", 0.70, True),
    "lapsed":      ("Contract period ended – check on site", "gray", 0.50, True),
    "announced":   ("Announced – opening not confirmed", "orange", 0.40, True),
    "listing":     ("Directory listing only", "gray", 0.35, True),
    "conflicting": ("Conflicting reports", "orange", 0.20, True),
    "restricted":  ("Restricted access", "violet", 0.00, False),
    "closed":      ("Non-operational", "red", 0.00, False),
}

SOURCE_KINDS = {"official": "Official document", "news": "News report", "wikipedia": "Wikipedia", "directory": "Third-party directory"}


class PuneDataError(ValueError):
    """Raised when a Pune data file breaks the validation rules."""


@dataclass(frozen=True)
class PuneData:
    facilities: pd.DataFrame
    destinations: pd.DataFrame
    sources: dict
    tariffs: dict
    policy: dict
    last_checked: str
    problems: list = field(default_factory=list)


def _read(name: str) -> dict:
    path = PUNE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. It ships with the repository; restore it with git.")
    return json.loads(path.read_text(encoding="utf-8"))


def _in_pune(lat, lon) -> bool:
    return PUNE_BBOX["lat"][0] <= lat <= PUNE_BBOX["lat"][1] and PUNE_BBOX["lon"][0] <= lon <= PUNE_BBOX["lon"][1]


def validate(facilities: list, destinations: list, sources: dict, tariffs: dict, policy: dict) -> list[str]:
    """Return a list of problems; an empty list means the data is valid."""
    problems = []
    ids = set()
    for f in facilities:
        fid = f.get("id", "?")
        if fid in ids:
            problems.append(f"{fid}: duplicate id")
        ids.add(fid)
        for key in ("name", "locality", "facility_type", "status", "status_note", "sources"):
            if not f.get(key):
                problems.append(f"{fid}: missing '{key}'")
        if f.get("status") not in STATUS:
            problems.append(f"{fid}: unknown status '{f.get('status')}'")
        if f.get("facility_type") not in FACILITY_TYPES:
            problems.append(f"{fid}: unknown facility_type '{f.get('facility_type')}'")
        for v in f.get("vehicle_types", []):
            if v not in VEHICLES:
                problems.append(f"{fid}: unknown vehicle type '{v}'")
        for s in f.get("sources", []):
            if s not in sources:
                problems.append(f"{fid}: source '{s}' not in sources.json")
        has_lat, has_lon = f.get("lat") is not None, f.get("lon") is not None
        if has_lat != has_lon:
            problems.append(f"{fid}: lat/lon must both be present or both null")
        if has_lat:
            if not _in_pune(f["lat"], f["lon"]):
                problems.append(f"{fid}: coordinates {f['lat']},{f['lon']} are outside the Pune area")
            if f.get("coord_source") not in sources:
                problems.append(f"{fid}: coordinates without a valid coord_source")
        if f.get("tariff_id") and f["tariff_id"] not in tariffs:
            problems.append(f"{fid}: tariff '{f['tariff_id']}' not defined")
    for t_id, t in tariffs.items():
        if t.get("source") not in sources:
            problems.append(f"tariff {t_id}: source missing")
        for veh, bands in t.get("bands", {}).items():
            if veh not in VEHICLES:
                problems.append(f"tariff {t_id}: unknown vehicle '{veh}'")
            limits = [b[0] for b in bands if b[0] is not None]
            if limits != sorted(limits):
                problems.append(f"tariff {t_id}/{veh}: bands not in increasing order")
    for d in destinations:
        if d.get("source") not in sources:
            problems.append(f"destination {d.get('id')}: source missing")
        if d.get("lat") is not None and not _in_pune(d["lat"], d["lon"]):
            problems.append(f"destination {d.get('id')}: outside the Pune area")
    for item in policy.get("items", []):
        if item.get("source") not in sources:
            problems.append(f"policy '{item.get('title')}': source missing")
    return problems


@lru_cache(maxsize=1)
def load_pune() -> PuneData:
    fac_raw = _read("facilities.json")
    dest_raw = _read("destinations.json")
    src_raw = _read("sources.json")
    pol_raw = _read("policy.json")
    sources = {s["id"]: s for s in src_raw["sources"]}
    tariffs = fac_raw.get("tariffs", {})
    problems = validate(fac_raw["facilities"], dest_raw["destinations"], sources, tariffs, pol_raw)
    if problems:
        raise PuneDataError("Pune data failed validation:\n- " + "\n- ".join(problems))

    fac = pd.DataFrame(fac_raw["facilities"])
    # Keep "not stated" as None (not NaN) so text checks like `if row["hours"]` behave.
    for col in ("operator", "hours", "address", "tariff_id", "coord_source", "capacity"):
        fac[col] = fac[col].astype(object).where(fac[col].notna(), None)
    fac["status_label"] = fac["status"].map(lambda s: STATUS[s][0])
    fac["status_color"] = fac["status"].map(lambda s: STATUS[s][1])
    fac["evidence"] = fac["status"].map(lambda s: STATUS[s][2])
    fac["recommendable"] = fac["status"].map(lambda s: STATUS[s][3])
    fac["type_label"] = fac["facility_type"].map(FACILITY_TYPES)
    fac["has_location"] = fac["lat"].notna()
    fac["has_tariff"] = fac["tariff_id"].notna()
    fac["vehicles_label"] = fac["vehicle_types"].map(lambda v: ", ".join(VEHICLES[x] for x in v) if v else "Not stated")
    fac["best_source_kind"] = fac["sources"].map(lambda ss: _best_kind(ss, sources))
    dest = pd.DataFrame(dest_raw["destinations"])
    return PuneData(facilities=fac, destinations=dest, sources=sources, tariffs=tariffs, policy=pol_raw,
                    last_checked=fac_raw.get("last_checked", "unknown"))


def _best_kind(source_ids, sources) -> str:
    order = ["official", "news", "wikipedia", "directory"]
    kinds = [sources[s]["kind"] for s in source_ids]
    return next((SOURCE_KINDS[k] for k in order if k in kinds), "Unknown")


def summary_counts(data: PuneData) -> dict:
    f = data.facilities
    return {
        "facilities": int(len(f)),
        "with_location": int(f["has_location"].sum()),
        "reported_operational": int((f["status"] == "reported").sum()),
        "announced_or_conflicting": int(f["status"].isin(["announced", "conflicting"]).sum()),
        "with_published_tariff": int(f["has_tariff"].sum()),
        "with_capacity": int(f["capacity"].notna().sum()),
        "sources": len(data.sources),
        "destinations": int(len(data.destinations)),
    }
