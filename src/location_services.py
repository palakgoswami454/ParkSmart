"""Distances, destination search and map links.

Only straight-line (great-circle) distance is computed. Driving distance would
need a routing service, which ParkSmart does not call, so every distance shown
in the app is labelled "straight-line".
"""
from __future__ import annotations

import math
from urllib.parse import quote_plus

import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def add_distances(facilities: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    out = facilities.copy()
    out["distance_km"] = [
        haversine_km(lat, lon, r.lat, r.lon) if pd.notna(r.lat) else float("nan") for r in out.itertuples()
    ]
    return out


def search_destinations(destinations: pd.DataFrame, query: str) -> pd.DataFrame:
    """Case-insensitive match on destination name or aliases."""
    q = (query or "").strip().lower()
    if not q:
        return destinations
    mask = destinations.apply(
        lambda r: q in r["name"].lower() or any(q in a or a in q for a in r.get("aliases", [])), axis=1)
    return destinations[mask]


def search_facilities(facilities: pd.DataFrame, query: str) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return facilities
    hay = (facilities["name"] + " " + facilities["locality"] + " " + facilities["address"].fillna("") + " "
           + facilities["type_label"]).str.lower()
    return facilities[hay.str.contains(q, regex=False)]


def directions_url(name: str, lat=None, lon=None, address: str | None = None) -> str:
    """Google Maps link (documented 'Maps URLs' scheme, no API key needed).

    With verified coordinates we link to directions to that point; otherwise we
    link to a text search for the name and address, so we never invent a pin.
    """
    if lat is not None and lon is not None and not (isinstance(lat, float) and math.isnan(lat)):
        return f"https://www.google.com/maps/dir/?api=1&destination={lat:.6f},{lon:.6f}"
    query = ", ".join(x for x in (name, address, "Pune") if x)
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


def walking_minutes(distance_km: float, speed_kmh: float = 4.5) -> float:
    """Rough walking time for a straight-line distance (real routes are longer)."""
    return distance_km / speed_kmh * 60
