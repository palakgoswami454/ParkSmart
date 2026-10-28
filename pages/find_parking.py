from datetime import datetime, time, timedelta

import pandas as pd
import streamlit as st

from src.data_loader import VEHICLES
from src.location_services import directions_url, walking_minutes
from src.recommendations import W_EVIDENCE, W_PROXIMITY, recommend
from src.ui import plain_map, guarded, page_header, pretty_date, pune, stay_cost_text, status_badge, theme_type
from src.visualizations import facility_map

# Dated context that matters for particular destinations (each item cites policy.json / sources.json).
DESTINATION_NOTES = {
    "airport": ("airport-15min", "Private vehicles may stay at most 15 minutes in the terminal area (₹500 fine, "
                                 "reported Jun 2026). Use the multi-level car park for longer stops."),
    "camp": ("pcb-camp", "Cantonment Board rates (Jul 2026): commuters reported being charged more than the official "
                         "rate, so check the board on site."),
    "pune-station": ("cr-paypark", "The station's pay-and-park contracts listed by Central Railway ended in "
                                   "Sep–Dec 2025; the current operator and rates were not verified."),
    "fc-road": ("pmc-pilot-2026", "PMC plans paid on-street parking on FC Road (tender stage, Mar 2026). "
                                  "Kerbside rules may change."),
    "jm-road": ("pmc-pilot-2026", "PMC plans paid on-street parking on JM Road (tender stage, Mar 2026)."),
    "laxmi-road": ("pmc-pilot-2026", "PMC plans paid on-street parking on Laxmi Road (tender stage, Mar 2026); "
                                     "more parking is being explored near Mandai (Jun 2026)."),
    "viman-nagar": ("pmc-pilot-2026", "PMC plans paid on-street parking in Viman Nagar (tender stage, Mar 2026)."),
    "baner": ("pmc-pilot-2026", "PMC plans paid on-street parking on Balewadi High Street (tender stage, Mar 2026)."),
    "pimpri": ("metro-pcmc-none", "Pune Metro was reported in Sep 2025 as saying no parking will be provided at "
                                  "metro stations in the PCMC area."),
    "swargate": ("metro-pmc-plan", "Swargate bus-stand parking was reported (Dec 2024) to be for bus travellers only."),
}


def page():
    data = pune()
    page_header("Find parking", "Tell ParkSmart where you're going. It ranks documented facilities by straight-line "
                                "distance and how well their status is evidenced, and lists what you'd still need "
                                "to check.", "pune")
    dests = data.destinations.set_index("id")
    qp = st.query_params
    ids = list(dests.index)
    default_dest = ids.index(qp["dest"]) if qp.get("dest") in ids else 0
    default_vehicle = qp.get("vehicle") if qp.get("vehicle") in VEHICLES else "car"

    with st.form("find"):
        c1, c2 = st.columns([2, 1.4])
        dest_id = c1.selectbox("Destination", ids, index=default_dest, format_func=lambda i: dests.loc[i, "name"])
        vehicle = c2.segmented_control("Vehicle", list(VEHICLES), default=default_vehicle, format_func=VEHICLES.get,
                                       required=True)
        c3, c4, c5, c6 = st.columns(4)
        now = datetime.now()
        arrive_date = c3.date_input("Arrival date", value=now.date())
        arrive_time = c4.time_input("Arrival time", value=time(min(now.hour + 1, 23), 0), step=timedelta(minutes=15))
        hours = c5.number_input("Stay (hours)", min_value=0.5, max_value=24.0, value=2.0, step=0.5)
        radius = c6.slider("Search radius (km)", 1.0, 10.0, 5.0, 0.5)
        b1, b2 = st.columns([1, 3], vertical_alignment="bottom")
        use_budget = b1.toggle("Set a budget")
        budget = b2.number_input("Maximum price for the stay (₹)", min_value=0, max_value=5000, value=100, step=10,
                                 disabled=not use_budget)
        submitted = st.form_submit_button("Show parking options", type="primary", icon=":material/near_me:")

    if not submitted and "dest" not in qp:
        st.info("Choose a destination and press **Show parking options**.", icon=":material/near_me:")
        return
    st.query_params.update({"dest": dest_id, "vehicle": vehicle})

    dest = dests.loc[dest_id].to_dict() | {"id": dest_id}
    rec = recommend(data, dest, vehicle, float(hours), float(budget) if use_budget else None, float(radius))
    arrival = datetime.combine(arrive_date, arrive_time)
    if arrival < datetime.now() - timedelta(minutes=5):
        st.warning("That arrival time is in the past. Results don't depend on time (there is no Pune occupancy data), "
                   "but check the date.", icon=":material/schedule:")

    st.markdown(f"### Parking near {dest['name']}")
    st.markdown(f"{VEHICLES[vehicle]} · arriving {arrival:%a %d %b, %H:%M} · staying {hours:g} h"
                + (f" · budget ₹{budget}" if use_budget else ""))
    with st.container(border=True):
        st.markdown(
            f"**What you can expect on arrival.** ParkSmart has **no occupancy data for Pune**, so it can't say how "
            f"full any of these will be at {arrival:%H:%M}. Rankings use only distance and documented status. Treat "
            "the top results as places worth checking, not guaranteed spaces.")
        if dest_id in DESTINATION_NOTES:
            src, text = DESTINATION_NOTES[dest_id]
            s = data.sources[src]
            st.markdown(f":material/campaign: {text} ([{s['publisher']}, {pretty_date(s['published'])}]({s['url']}))")

    has_point = dest.get("lat") is not None and not pd.isna(dest.get("lat"))
    if has_point:
        st.plotly_chart(facility_map(pd.concat([rec.ranked, rec.not_suitable[rec.not_suitable["has_location"]]]),
                                     theme_type(), offline=plain_map(), height=380, destination=dest, radius_km=radius),
                        width="stretch", theme="streamlit", config={"displayModeBar": False, "scrollZoom": True})
    else:
        st.caption("This destination has no verified map point, so options are matched by area name only.")

    if rec.ranked.empty and rec.no_location.empty:
        st.warning(f"No documented facility within {radius:g} km fits these choices. Widen the radius, change the "
                   "vehicle or budget, or see the list of unsuitable options below.", icon=":material/search_off:")

    for i, r in rec.ranked.head(6).iterrows():
        with st.container(border=True):
            top = st.columns([0.35, 3, 1.4], vertical_alignment="center")
            top[0].markdown(f"### {i + 1}")
            top[1].markdown(f"**{r['name']}**  \n{r['type_label']} · {r['locality']}")
            status_badge(r, top[2])
            c = st.columns([1.2, 1.6, 1.4, 1.2])
            c[0].metric("Straight-line", f"{r['distance_km']:.1f} km",
                        help=f"About {walking_minutes(r['distance_km']):.0f} min on foot in a straight line; real "
                             "routes are longer. Driving distance is not calculated.")
            c[1].markdown("**Cost for your stay**  \n" + stay_cost_text(r, data, vehicle, float(hours)))
            c[2].markdown(f"**Score {r['score']:.2f}**  \nproximity {r['proximity']:.2f} · evidence {r['evidence']:.2f}")
            c[3].link_button("Directions", directions_url(r["name"], r["lat"], r["lon"]), icon=":material/directions:",
                             width="stretch")
            st.caption(f"{r['status_note']}  \n**Still unknown:** {r['missing']}.")
            if st.button("Facility details", key=f"det-{r['id']}", icon=":material/info:"):
                st.switch_page("pages/explore.py", query_params={"facility": r["id"]})

    if not rec.no_location.empty:
        st.markdown("#### Also in this area (no verified map location)")
        for _, r in rec.no_location.iterrows():
            with st.container(border=True):
                a, b, c = st.columns([3, 1.4, 1.2], vertical_alignment="center")
                a.markdown(f"**{r['name']}** · {r['locality']}  \n{stay_cost_text(r, data, vehicle, float(hours))}")
                status_badge(r, b)
                c.link_button("Search in Maps", directions_url(r["name"], address=r["address"]), width="stretch")

    if not rec.not_suitable.empty:
        with st.expander(f"Not suitable for this trip ({len(rec.not_suitable)})"):
            st.dataframe(rec.not_suitable[["name", "distance_km", "reason"]].rename(
                columns={"name": "Facility", "distance_km": "Straight-line km", "reason": "Why it's excluded"}),
                hide_index=True, width="stretch", column_config={"Straight-line km": st.column_config.NumberColumn(format="%.1f")})

    with st.expander("How the ranking works"):
        st.markdown(f"""
**Score = {W_PROXIMITY} × proximity + {W_EVIDENCE} × evidence**

- *Proximity* = 1 − straight-line distance ÷ search radius (0 at the edge of the radius).
- *Evidence* reflects how well the facility's status is documented: reported operational 0.90 · private venue 0.70 ·
  contract period ended 0.50 · announced 0.40 · directory listing 0.35 · conflicting reports 0.20.
- Facilities are **excluded** only for a known reason: non-operational or restricted, a vehicle type the sources don't
  list, a published price above your budget, or outside the radius. Missing information never excludes a facility; it
  is listed under *Still unknown*.
- Distances are great-circle distances between cited coordinates (stations, landmarks). They are not driving
  distances and parking entrances may be a few hundred metres from the pin.
- **Occupancy is not part of the score** because no Pune occupancy data exists.
""")


guarded(page)
