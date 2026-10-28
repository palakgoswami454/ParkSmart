import streamlit as st

from src.data_loader import VEHICLES, summary_counts
from src.ui import (plain_map, MissingArtifact, grid, guarded, metrics_json, page_header, pretty_date, pune, status_badge,
                    theme_type)
from src.visualizations import daily_trend, facility_map, hour_profile


def page():
    data = pune()
    fac = data.facilities
    page_header("Where should I park in Pune?",
                "Pick a destination to compare documented parking options and see exactly what is known about each "
                "one, and what isn't.", "mixed")

    with st.container(border=True):
        c1, c2, c3 = st.columns([2.4, 1.3, 0.9], vertical_alignment="bottom")
        names = dict(zip(data.destinations["id"], data.destinations["name"]))
        dest = c1.selectbox("Destination", list(names), index=None, format_func=names.get,
                            placeholder="Type FC Road, Pune Station, Hinjawadi…")
        vehicle = c2.segmented_control("Vehicle", ["two_wheeler", "car"], default="car", format_func=VEHICLES.get,
                                       required=True)
        if c3.button("Find parking", type="primary", icon=":material/near_me:", disabled=dest is None,
                     width="stretch"):
            st.switch_page("pages/find_parking.py", query_params={"dest": dest, "vehicle": vehicle})

    n = summary_counts(data)
    m = st.columns(5)
    m[0].metric("Facilities in directory", n["facilities"], help="Every entry has at least one source link.")
    m[1].metric("Shown on the map", n["with_location"], help="Entries with coordinates from a cited source.")
    m[2].metric("Reported operational", n["reported_operational"],
                help="Status backed by a dated report or official statement. Not a live check.")
    m[3].metric("With published charges", n["with_published_tariff"])
    m[4].metric("Sources cited", n["sources"])

    left, right = st.columns([1.75, 1], gap="large")
    with left:
        st.plotly_chart(facility_map(fac, theme_type(), offline=plain_map(), height=470), width="stretch", theme="streamlit",
                        config={"displayModeBar": False, "scrollZoom": True})
        st.caption("Colour = how well the facility's status is documented, not how full it is. "
                   "Pins sit at the cited station or landmark; parking entrances are usually not published. "
                   "Map © CARTO, © OpenStreetMap contributors.")
    with right:
        st.subheader("Best-documented options")
        for _, r in fac[fac["status"] == "reported"].iterrows():
            with st.container(border=True):
                a, b = st.columns([3, 1.1], vertical_alignment="center")
                a.markdown(f"**{r['name']}**  \n{r['type_label']} · {r['locality']}")
                if b.button("Details", key=f"ov-{r['id']}", width="stretch"):
                    st.switch_page("pages/explore.py", query_params={"facility": r["id"]})
        k = fac["status"].isin(["announced", "conflicting"]).sum()
        st.markdown(f"**{k} more** are announced or have conflicting reports, mostly the metro-station lots "
                    "announced in Feb 2024. See them on *Explore parking*.")

    with st.container(border=True):
        st.markdown("**What's live, what's historical, what's missing**")
        st.markdown(
            f"- **Facility facts** come from cited documents and news, last checked {pretty_date(data.last_checked)}. "
            "Status can have changed since.\n"
            "- **No live availability.** No public occupancy feed or history was found for any Pune car park, so "
            "ParkSmart never claims a Pune facility has free spaces.\n"
            "- **The forecasting model** was trained and tested on Birmingham, UK (2016). It is demonstrated on that "
            "data and can be tested on your own counts, but it is not applied to Pune facilities.")

    st.subheader("Occupancy patterns in the model's training data")
    st.caption("Birmingham City Council car parks, 4 Oct – 19 Dec 2016. Shown because the forecasting model learned "
               "from it. These are not Pune patterns.")
    try:
        g = grid()
        ev = metrics_json("evaluation_summary.json")
        c1, c2 = st.columns(2)
        c1.markdown("**Average day**")
        c1.plotly_chart(hour_profile(g), width="stretch", theme="streamlit", config={"displayModeBar": False})
        c2.markdown("**Daily mean over the study period**")
        c2.plotly_chart(daily_trend(g), width="stretch", theme="streamlit", config={"displayModeBar": False})
        sel = [r for r in ev["60"]["regression_table"] if r["Selected"]][0]
        st.markdown(f"On the untouched test fortnight, the selected model forecast occupancy 60 minutes ahead with an "
                    f"average error of **{sel['Test MAE (pp)']:.2f} percentage points (≈ {sel['Test MAE (spaces)']:.0f} "
                    f"spaces)**, against {[r for r in ev['60']['regression_table'] if r['Model'] == 'Persistence (baseline)'][0]['Test MAE (pp)']:.2f} "
                    "for assuming nothing changes.")
    except MissingArtifact as e:
        st.warning(f"Training-data charts unavailable: {e}")


guarded(page)
