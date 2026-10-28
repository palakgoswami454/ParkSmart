"""Reusable Streamlit building blocks: cached data access, badges, source lists,
facility cards and error handling. Pages import from here so every page shows
data, provenance and failures the same way."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config as C
from src.data_loader import SOURCE_KINDS, STATUS, VEHICLES, PuneData, load_pune
from src.location_services import directions_url
from src.recommendations import tariff_cost


class MissingArtifact(Exception):
    """A generated file the page needs is missing; message says how to create it."""


FIX_COMMANDS = ("python -m src.data_preprocessing\npython -m src.eda\npython -m src.train\n"
                "python -m src.intervals\npython -m src.evaluate\npython -m src.transfer")


def _need(path: Path, how: str):
    if not path.exists():
        raise MissingArtifact(f"`{path.relative_to(C.ROOT)}` is missing. {how}")


# ----------------------------------------------------------------------------
# Cached data access
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def pune() -> PuneData:
    return load_pune()


@st.cache_data(show_spinner="Loading the Birmingham occupancy grid…")
def grid() -> pd.DataFrame:
    _need(C.GRID_CSV, "Run `python -m src.data_preprocessing`.")
    from src.data_preprocessing import load_grid
    return load_grid()


@st.cache_resource(show_spinner="Loading the trained model…")
def bundle(h: int) -> dict:
    from src.predict import ModelNotFoundError, load_bundle
    try:
        return load_bundle(h)
    except ModelNotFoundError as e:
        raise MissingArtifact(str(e)) from e


@st.cache_data(show_spinner="Running the model over the replay data…")
def inference(h: int) -> pd.DataFrame:
    from src.predict import inference_table
    return inference_table(grid(), bundle(h))


@st.cache_data
def metrics_json(name: str, how: str = "Run `python -m src.train` and `python -m src.evaluate`.") -> dict:
    p = C.METRICS_DIR / name
    _need(p, how)
    return json.loads(p.read_text())


def theme_type() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:  # older Streamlit or no browser context
        return "light"


def plain_map() -> bool:
    """True when the user switched the map to the offline (plain) background."""
    return not st.session_state.get("street_map", True)


def guarded(page):
    """Run a page; turn missing generated files into an actionable message."""
    try:
        page()
    except MissingArtifact as e:
        st.error(f"This page needs files that haven't been generated yet. {e}", icon=":material/build:")
        st.code(FIX_COMMANDS, language="bash")


# ----------------------------------------------------------------------------
# Small presentational helpers
# ----------------------------------------------------------------------------
def page_header(title: str, subtitle: str, mode: str):
    """mode: 'pune' (sourced directory), 'model' (Birmingham replay) or 'mixed'."""
    st.title(title)
    st.markdown(subtitle)
    chips = {
        "pune": ("Pune directory · sources checked " + pretty_date(pune().last_checked), "blue", ":material/verified:"),
        "model": ("Historical model · Birmingham, UK, Oct–Dec 2016", "orange", ":material/history:"),
        "mixed": ("Pune directory + Birmingham-trained model, kept separate", "gray", ":material/call_split:"),
    }
    label, color, icon = chips[mode]
    st.badge(label, color=color, icon=icon)


def birmingham_notice():
    st.info("**This is not Pune data.** Forecasts on this page come from a model trained and tested on "
            "Birmingham City Council car-park counts (4 Oct – 19 Dec 2016). The app replays that history: you pick a "
            "moment, the model sees only data up to it, and the real outcome is shown for comparison. Nothing here is a "
            "live reading.", icon=":material/info:")


def status_badge(row, container=st):
    label, color, _, _ = STATUS[row["status"]]
    container.badge(label, color=color)


def pretty_date(s) -> str:
    if not s:
        return "date not given"
    try:
        if len(str(s)) == 7:
            return pd.Timestamp(str(s) + "-01").strftime("%b %Y")
        t = pd.Timestamp(str(s))
        return f"{t.day} {t:%b %Y}"  # no %-d: it is not supported on Windows
    except Exception:
        return str(s)


def source_line(src: dict) -> str:
    when = f"published {pretty_date(src['published'])}" if src.get("published") else "undated"
    return (f"[{src['title']}]({src['url']}) — {src['publisher']}, {when}; "
            f"{SOURCE_KINDS[src['kind']].lower()}, accessed {pretty_date(src['accessed'])}")


def sources_list(ids, data: PuneData):
    st.markdown("\n".join(f"- {source_line(data.sources[s])}" for s in ids))


def tariff_table(tariff: dict) -> pd.DataFrame:
    rows = []
    for veh, bands in tariff["bands"].items():
        prev = 0
        for limit, price in bands:
            if limit is None:
                band = f"over {prev} h"
            elif prev == 0:
                band = f"up to {limit} h"
            elif limit == 24:
                band = "full day"
            else:
                band = f"{prev}–{limit} h"
            rows.append({"Vehicle": VEHICLES[veh], "Stay": band, "Price (₹)": price})
            prev = limit
    return pd.DataFrame(rows)


def capacity_text(cap) -> str:
    if not isinstance(cap, dict):
        return "Not published"
    if "total" in cap:
        return f"{cap['total']:,}{'+' if cap.get('is_lower_bound') else ''} spaces"
    parts = [f"{v:,} {VEHICLES[k].lower()}s" for k, v in cap.items() if k in VEHICLES]
    return ", ".join(parts)


def facility_detail(row, data: PuneData):
    """Full facility card used by Explore Parking."""
    with st.container(border=True):
        top = st.columns([3, 1.3], vertical_alignment="center")
        top[0].subheader(row["name"])
        top[0].caption(f"{row['type_label']} · {row['locality']}")
        status_badge(row, top[1])
        st.markdown(row["status_note"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Address**  \n" + (row["address"] or "Not published in the sources found"))
            st.markdown("**Operator**  \n" + (row["operator"] or "Not stated"))
            st.markdown("**Vehicles**  \n" + row["vehicles_label"])
            st.markdown("**Capacity**  \n" + capacity_text(row["capacity"]))
            st.markdown("**Hours**  \n" + (row["hours"] or "Not published"))
        with c2:
            loc = row["coord_precision"]
            st.markdown("**Location on map**  \n" + loc)
            url = directions_url(row["name"], row["lat"] if row["has_location"] else None,
                                 row["lon"] if row["has_location"] else None, row["address"])
            st.link_button("Open directions in Google Maps" if row["has_location"] else "Search in Google Maps",
                           url, icon=":material/directions:")
            if row["tariff_id"]:
                t = data.tariffs[row["tariff_id"]]
                st.markdown(f"**Published charges** · {t['label']}")
                st.dataframe(tariff_table(t), hide_index=True, width="stretch")
                st.caption(t["notes"])
            else:
                st.markdown("**Charges**  \nNo published tariff found. Check the rate board on site.")

        with st.container(border=True):
            st.markdown("**Occupancy forecast**")
            st.markdown(
                "Not available. No public occupancy feed or history exists for this facility, and ParkSmart's model "
                "needs the facility's current count to forecast the next 30–60 minutes. It was trained on Birmingham "
                "car parks and has not been validated in Pune.")
            if isinstance(row["capacity"], dict):
                st.caption("Capacity is published, so free spaces could be estimated (capacity − predicted occupied) "
                           "once occupancy counts become available.")
            st.page_link("pages/forecast.py", label="See how the forecast works, or test it on your own counts",
                         icon=":material/insights:")

        st.markdown("**Sources**")
        sources_list(row["sources"], data)
        st.caption(f"Directory last checked {pretty_date(data.last_checked)}. Coordinates are from "
                   f"{data.sources[row['coord_source']]['publisher'] if row['coord_source'] else 'no source'}.")


def stay_cost_text(row, data: PuneData, vehicle: str, hours: float) -> str:
    t = data.tariffs.get(row["tariff_id"]) if row["tariff_id"] else None
    price, note = tariff_cost(t, vehicle, hours)
    if price is None:
        return note
    return f"₹{price:.0f} ({note}, tariff of {pretty_date(t['as_of'])})"


def today() -> date:
    return date.today()
