"""Click a location -> predicted Safia sales index (1-100), range and reasons.

Run:  uv run streamlit run app/app.py
"""
import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import load  # noqa: E402

st.set_page_config(page_title="The Next Safia", layout="wide")


@st.cache_resource
def model():
    return load()


@st.cache_data
def data():
    b = pd.read_csv(ROOT / "data/processed/branches.csv")
    g = ROOT / "data/processed/tashkent_grid.parquet"
    return b, (pd.read_parquet(g) if g.exists() else None)


sm = model()
branches, grid = data()
if "pt" not in st.session_state:
    st.session_state.pt = (41.3111, 69.2797)

st.title("The Next Safia: location potential")
left, right = st.columns([3, 2])

with right:
    st.subheader("Location")
    lat = st.number_input("Latitude", value=float(st.session_state.pt[0]), format="%.6f")
    lon = st.number_input("Longitude", value=float(st.session_state.pt[1]), format="%.6f")
    st.caption("…or click anywhere on the map.")
    res, F = sm.predict(lat, lon)
    r = res.iloc[0]
    st.metric("Predicted sales index (1–100)", f"{r.sales_index}", help="Same scale as the supplied file")
    st.write(f"**80% range:** {r.low_80} – {r.high_80}")
    st.write(f"**Confidence:** {r.confidence} (nearest training branch {r.nearest_known_branch_km} km)")
    pct = (branches.sales < r.sales_index).mean() * 100
    st.write(f"Better than about **{pct:.0f}%** of current branches (network median {branches.sales.median():.1f}).")
    st.subheader("Why (vs. a typical branch)")
    for d in r.drivers.split("; "):
        if d:
            st.write("• " + d)
    with st.expander("All features for this point"):
        st.dataframe(F.T.rename(columns={0: "value"}))

with left:
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="cartodbpositron")
    if grid is not None and st.checkbox("Show predicted-potential heatmap (Tashkent)", value=True):
        HeatMap(grid[["lat", "lon", "sales_index"]].values.tolist(), radius=18, blur=22, min_opacity=0.2,
                max_val=float(grid.sales_index.quantile(0.98))).add_to(m)
    for la, lo in sm.fz.safia_net[["lat", "lon"]].values:
        folium.CircleMarker([la, lo], radius=2, color="#666", fill=True).add_to(m)
    for _, b in branches.iterrows():
        folium.CircleMarker([b.lat, b.lon], radius=3 + b.sales / 10, color="#c0392b", fill=True, weight=1,
                            tooltip=f"{b['name']}: {b.sales:.1f}").add_to(m)
    folium.Marker([lat, lon], icon=folium.Icon(color="green", icon="star")).add_to(m)
    out = st_folium(m, height=650, use_container_width=True, returned_objects=["last_clicked"])
    if out and out.get("last_clicked"):
        c = out["last_clicked"]
        if (round(c["lat"], 6), round(c["lng"], 6)) != tuple(round(x, 6) for x in st.session_state.pt):
            st.session_state.pt = (c["lat"], c["lng"])
            st.rerun()
