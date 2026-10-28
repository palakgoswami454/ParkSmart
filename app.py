"""ParkSmart — Pune parking discovery and a separately documented occupancy-forecasting model.

Run:  streamlit run app.py

Pages live in pages/. Pune facts come from data/pune/*.json (every fact has a
source). Forecasts come from the model trained on Birmingham 2016 data and are
only ever shown for Birmingham car parks or for occupancy counts the user
uploads; they are never attached to a Pune facility.
"""
from __future__ import annotations

import streamlit as st

from src import config as C
from src.ui import pretty_date, pune

st.set_page_config(page_title="ParkSmart · Pune", page_icon="assets/icon.svg", layout="wide",
                   initial_sidebar_state="expanded")
st.logo("assets/logo.svg", icon_image="assets/icon.svg", size="large")

pages = {
    "Pune": [
        st.Page("pages/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page("pages/explore.py", title="Explore parking", icon=":material/map:", url_path="explore"),
        st.Page("pages/find_parking.py", title="Find parking", icon=":material/near_me:", url_path="find"),
        st.Page("pages/city_insights.py", title="City insights", icon=":material/location_city:", url_path="insights"),
    ],
    "Forecasting model": [
        st.Page("pages/forecast.py", title="Occupancy forecast", icon=":material/insights:", url_path="forecast"),
        st.Page("pages/model_performance.py", title="Model performance", icon=":material/analytics:", url_path="models"),
    ],
    "About": [
        st.Page("pages/about.py", title="Data sources", icon=":material/fact_check:", url_path="about"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown(":material/location_on: **Pune, Maharashtra**")
    try:
        checked = pretty_date(pune().last_checked)
    except Exception:  # data error is reported on the page itself
        checked = "unknown"
    with st.container(border=True):
        st.markdown("**Data mode**")
        st.markdown(f":blue-badge[Directory] sourced facts, checked {checked}")
        st.markdown(":orange-badge[Historical model] Birmingham 2016 replay")
        st.markdown(":gray-badge[Live data] not connected")
    if "street_map" not in st.session_state:
        st.session_state["street_map"] = st.query_params.get("basemap") != "plain"
    st.toggle("Street map", key="street_map", help="Street tiles load from CARTO and need internet. Turn off for a "
              "plain background that works offline (markers and details stay).")
    st.caption(":material/contrast: Light or dark: open the **⋮** menu (top right) → **Settings** → **Theme**. "
               "Your browser remembers the choice.")
    st.caption("AUCML ESE project · Symbiosis Institute of Technology, Pune")

nav.run()
