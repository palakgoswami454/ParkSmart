import pandas as pd
import streamlit as st

from src.data_loader import SOURCE_KINDS
from src.ui import guarded, page_header, pretty_date, pune


def page():
    data = pune()
    page_header("Data sources", "Where every number in ParkSmart comes from, and what it can't tell you.", "mixed")

    st.subheader("What each part of the app is")
    st.dataframe(pd.DataFrame([
        {"Shown in the app": "Pune facility names, addresses, operators, charges, capacities",
         "Kind": "Documented (dated sources)", "Live?": "No",
         "Where from": f"data/pune/*.json, checked {pretty_date(data.last_checked)}"},
        {"Shown in the app": "Pune facility status (reported, announced, conflicting…)",
         "Kind": "Evidence label assigned from the sources", "Live?": "No", "Where from": "data/pune/facilities.json"},
        {"Shown in the app": "Map pins and distances",
         "Kind": "Coordinates of cited stations/landmarks; straight-line distance", "Live?": "No",
         "Where from": "Wikipedia articles and one directory listing (see each facility)"},
        {"Shown in the app": "Pune occupancy, availability, forecasts", "Kind": "Unavailable", "Live?": "—",
         "Where from": "No public data found"},
        {"Shown in the app": "Birmingham occupancy (charts, replay)", "Kind": "Historical (2016)", "Live?": "No",
         "Where from": "UCI Parking Birmingham dataset"},
        {"Shown in the app": "Forecasts and 90% ranges", "Kind": "Predicted by the model", "Live?": "No",
         "Where from": "models/*.joblib trained on Birmingham"},
        {"Shown in the app": "Your uploaded counts", "Kind": "Your data, evaluated before use", "Live?": "No",
         "Where from": "Occupancy forecast → Test on your own data"},
    ]), hide_index=True, width="stretch")

    st.subheader("Model training data")
    st.markdown("**Parking Birmingham** — Stolfi, D. (2017), UCI Machine Learning Repository, "
                "[doi:10.24432/C51K5Z](https://doi.org/10.24432/C51K5Z), CC BY 4.0. 35,717 readings from 30 Birmingham "
                "City Council car parks, 4 Oct – 19 Dec 2016, about every 30 minutes from 08:00 to 16:30. "
                "See `data/README.md` for the audit and cleaning steps.")

    st.subheader("Pune sources")
    src = pd.DataFrame(data.sources.values())
    used = {s for ids in data.facilities["sources"] for s in ids} | set(data.destinations["source"]) | \
           {i["source"] for i in data.policy["items"]}
    src["Used for"] = src["id"].map(lambda i: "facilities / policy / map" if i in used else "—")
    kind = st.pills("Kind", list(SOURCE_KINDS), selection_mode="multi", default=list(SOURCE_KINDS),
                    format_func=SOURCE_KINDS.get)
    view = src[src["kind"].isin(kind or [])]
    st.dataframe(view.assign(kind=view["kind"].map(SOURCE_KINDS))[["title", "publisher", "kind", "published", "accessed",
                                                                   "url"]].rename(columns={
        "title": "Title", "publisher": "Publisher", "kind": "Kind", "published": "Published", "accessed": "Accessed",
        "url": "Link"}), hide_index=True, width="stretch",
        column_config={"Link": st.column_config.LinkColumn(display_text="open")})
    st.caption("Two links supplied with the project brief (the PMC open-data portal and the Pune Smart City planning "
               "document) could not be read from the build environment, so nothing in the app relies on them.")

    st.subheader("How facts were checked")
    st.markdown(
        "- A fact enters the directory only if a cited source states it. Empty fields mean the sources don't say.\n"
        "- Coordinates come from the cited article or listing. Locality articles with clearly wrong coordinates "
        "(Kothrud, Aundh, Pune Camp) were rejected and a nearby station used instead.\n"
        "- `src/data_loader.py` refuses to load the directory if a fact has no source, a coordinate is outside Pune, or "
        "a status or tariff is malformed; `tests/test_pune.py` runs those checks.\n"
        "- Statuses describe evidence on the source date. A facility marked *reported operational* may have changed since.")

    st.subheader("Limitations")
    st.markdown(
        "- No Pune occupancy data, so no Pune forecasts or availability.\n"
        "- Small directory (19 facilities) built from what could be verified; many real car parks are missing.\n"
        "- Straight-line distances only; no routing or traffic.\n"
        "- The model was trained on one Birmingham autumn and covers 08:00–16:30 only.\n"
        "- Map tiles load from CARTO in your browser and need an internet connection.")


guarded(page)
