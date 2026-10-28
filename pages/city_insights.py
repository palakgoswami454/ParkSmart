import pandas as pd
import streamlit as st

from src.ui import MissingArtifact, guarded, metrics_json, page_header, pretty_date, pune


def page():
    data = pune()
    pol = data.policy
    fac = data.facilities
    page_header("City insights", "Pune's parking rules, plans and gaps, each with its date and source. Dated "
                                 "information, not current conditions.", "pune")

    st.subheader("Metro-station parking: announced vs. reported")
    metro = fac[fac["facility_type"] == "metro"]
    c = st.columns(3)
    c[0].metric("Stations announced (Feb 2024)", int((metro["sources"].map(lambda s: "metro-rel-rw" in s)).sum()))
    c[1].metric("Later contradicted", int((metro["status"] == "conflicting").sum()),
                help="PCMC (Sep 2025 report) and Swargate (Dec 2024 report).")
    c[2].metric("Opening confirmed", 0, help="No report confirming that the announced lots opened was found.")
    st.markdown("Pune Metro announced parking at nine stations in February 2024. A December 2024 report said most "
                "stations still lacked designated parking and PMC set aside ₹5 crore to find space, and in September "
                "2025 Pune Metro was reported as ruling out parking in the PCMC area. Treat metro-station parking as "
                "*worth checking*, not as available.")

    st.subheader("Timeline")
    for item in sorted(pol["items"], key=lambda i: i["date"], reverse=True):
        s = data.sources[item["source"]]
        with st.container(border=True):
            a, b = st.columns([1, 5])
            a.markdown(f"**{pretty_date(item['date'])}**")
            a.badge(item["status"], color="gray")
            b.markdown(f"**{item['title']}**  \n{item['summary']}")
            b.caption(f"Source: [{s['title']}]({s['url']}), {s['publisher']}")

    st.subheader("What parking costs, on paper")
    a, b = st.columns(2, gap="large")
    z16 = pol["zone_rates_2016"]
    with a:
        st.markdown(f"**PMC policy base rates (2016)** · {z16['unit']}")
        st.dataframe(pd.DataFrame(z16["rows"], columns=z16["columns"]), hide_index=True, width="stretch")
        st.caption("Zone A = central business district, B = mobility corridors, C = rest of the city. Off-street is "
                   "set below on-street to push cars off the road.")
    z26 = pol["zone_rates_2026_proposed"]
    with b:
        st.markdown(f"**PMC proposal (Sep 2026)** · {z26['unit']}")
        st.dataframe(pd.DataFrame(z26["rows"], columns=z26["columns"]), hide_index=True, width="stretch")
        st.caption("Proposed only: needs Standing Committee and General Body approval.")
    st.markdown("**Published facility charges in the directory**")
    rows = []
    for t_id, t in data.tariffs.items():
        for veh, bands in t["bands"].items():
            rows.append({"Tariff": t["label"], "Vehicle": veh.replace("_", "-"),
                         "Shortest band (₹)": bands[0][1], "Longest band (₹)": bands[-1][1]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("What isn't public")
    st.markdown("\n".join(f"- {x}" for x in pol["searched_but_not_found"]))
    st.caption("Because of these gaps ParkSmart shows no occupancy, availability or forecast for any Pune facility.")

    st.subheader("Demand patterns the model learned (Birmingham, not Pune)")
    try:
        eda = metrics_json("eda_summary.json", "Run `python -m src.eda`.")
        ev = metrics_json("evaluation_summary.json")
        hp = pd.Series(eda["hour_profile_mean"])
        busy = hp[hp >= hp.max() - 0.03].index.tolist()
        e = ev["60"]
        st.markdown(
            f"- Birmingham car parks peaked at **{eda['peak_time']}** (average {eda['peak_mean_rate']:.0%} full) and "
            f"stayed near the peak from {busy[0]} to {busy[-1]}.\n"
            f"- Weekends averaged {eda['weekday_vs_weekend_mean']['weekend']:.0%} vs "
            f"{eda['weekday_vs_weekend_mean']['weekday']:.0%} on weekdays, but rose towards Christmas.\n"
            f"- Forecasts were hardest during the morning fill-up and at weekends "
            f"({e['segments']['weekend']['selected_MAE_pp']:.2f} pp vs {e['segments']['weekday']['selected_MAE_pp']:.2f} "
            "pp on weekdays).\n"
            "- Whether Pune follows the same daily shape is unknown until Pune counts are available.")
    except MissingArtifact as err:
        st.warning(str(err))


guarded(page)
