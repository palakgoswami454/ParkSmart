import streamlit as st

from src.data_loader import FACILITY_TYPES, STATUS, VEHICLES
from src.location_services import search_facilities
from src.ui import plain_map, facility_detail, guarded, page_header, pune, theme_type
from src.visualizations import facility_map


def page():
    data = pune()
    fac = data.facilities
    page_header("Explore parking",
                "Every parking facility ParkSmart could document in Pune, with its evidence and sources.", "pune")

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1.6, 1.2])
        q = c1.text_input("Search", placeholder="Name, area or type, e.g. metro, Swargate, airport")
        types = c2.multiselect("Facility type", list(FACILITY_TYPES), format_func=FACILITY_TYPES.get,
                               placeholder="All types")
        vehicle = c3.selectbox("Vehicle", ["any"] + list(VEHICLES),
                               format_func=lambda v: "Any vehicle" if v == "any" else VEHICLES[v])
        status_opts = list(STATUS)
        statuses = st.pills("Evidence status", status_opts, selection_mode="multi", default=status_opts,
                            format_func=lambda s: STATUS[s][0])
        t1, t2, _ = st.columns([1.2, 1.2, 2])
        only_priced = t1.toggle("Published charges only")
        only_mapped = t2.toggle("On the map only")

    view = search_facilities(fac, q)
    if types:
        view = view[view["facility_type"].isin(types)]
    view = view[view["status"].isin(statuses or [])]
    if vehicle != "any":
        # Facilities whose sources don't list vehicle types stay in the results (unknown ≠ excluded).
        view = view[view["vehicle_types"].map(lambda v: not v or vehicle in v)]
    if only_priced:
        view = view[view["has_tariff"]]
    if only_mapped:
        view = view[view["has_location"]]

    preselect = st.query_params.get("facility")
    if view.empty:
        st.warning("No facility matches these filters. Clear the search or select more evidence statuses.",
                   icon=":material/search_off:")
        return

    left, right = st.columns([1.1, 1.25], gap="large")
    with right:
        st.markdown(f"**{len(view)} of {len(fac)} facilities** · select a row for details")
        table = view[["name", "locality", "status_label"]].rename(
            columns={"name": "Facility", "locality": "Area", "status_label": "Status"})
        event = st.dataframe(table, hide_index=True, width="stretch", height=430, on_select="rerun",
                             selection_mode="single-row", key="fac_table")
    rows = event.selection.rows if event and event.selection else []
    if rows:
        selected = view.iloc[rows[0]]["id"]
    elif preselect in set(view["id"]):
        selected = preselect
    else:
        selected = None
    with left:
        st.plotly_chart(facility_map(view, theme_type(), offline=plain_map(), height=470, highlight=selected), width="stretch",
                        theme="streamlit", config={"displayModeBar": False, "scrollZoom": True})
        hidden = int((~view["has_location"]).sum())
        st.caption((f"{hidden} matching facilit{'y has' if hidden == 1 else 'ies have'} no verified coordinates and "
                    "appear only in the table. " if hidden else "") + "Map © CARTO, © OpenStreetMap contributors.")

    if selected:
        facility_detail(view[view["id"] == selected].iloc[0], data)
    else:
        st.info("Select a facility in the table to see its address, charges, directions and sources.",
                icon=":material/touch_app:")


guarded(page)
