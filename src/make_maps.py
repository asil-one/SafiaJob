"""Precompute a predicted-potential grid over Tashkent and write a standalone HTML map.

Output:
  data/processed/tashkent_grid.parquet   grid predictions (used by the app)
  reports/map.html                       open in any browser, no server needed
"""
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from folium.plugins import HeatMap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEP = 0.005  # ~450-550 m


def tashkent_grid(sm):
    lat = np.arange(41.20, 41.43, STEP)
    lon = np.arange(69.12, 69.42, STEP)
    LA, LO = np.meshgrid(lat, lon)
    pts = pd.DataFrame({"lat": LA.ravel(), "lon": LO.ravel()})
    F = sm.fz.transform(pts.lat.values, pts.lon.values)
    pts = pts[np.expm1(F["pop_1000"].values) > 2000].reset_index(drop=True)  # skip empty land
    res, _ = sm.predict(pts.lat.values, pts.lon.values, explain=False)
    return res


def build_map(grid, branches, net):
    m = folium.Map(location=[41.31, 69.27], zoom_start=12, tiles="cartodbpositron")
    cm = LinearColormap(["#2c7bb6", "#ffffbf", "#d7191c"], vmin=5, vmax=35, caption="Predicted sales index")
    HeatMap(grid[["lat", "lon", "sales_index"]].values.tolist(), name="Predicted potential",
            radius=18, blur=22, min_opacity=0.25, max_val=float(grid.sales_index.quantile(0.98))).add_to(m)
    fg = folium.FeatureGroup("Known Safia (2GIS+OSM+file)", show=True)
    for la, lo in net[["lat", "lon"]].values:
        folium.CircleMarker([la, lo], radius=2, color="#555", fill=True, opacity=0.6).add_to(fg)
    fg.add_to(m)
    fb = folium.FeatureGroup("Training branches (actual sales)", show=True)
    for _, r in branches.iterrows():
        folium.CircleMarker([r.lat, r.lon], radius=4 + r.sales / 8, color=cm(min(r.sales, 35)), fill=True,
                            fill_opacity=0.9, weight=1,
                            tooltip=f"{r['name']} ({r.code}): actual {r.sales:.1f}").add_to(fb)
    fb.add_to(m)
    top = grid.nlargest(15, "sales_index")
    ft = folium.FeatureGroup("Top-15 predicted grid cells", show=True)
    for _, r in top.iterrows():
        folium.Marker([r.lat, r.lon], icon=folium.Icon(color="green", icon="star"),
                      tooltip=f"predicted {r.sales_index} ({r.low_80}-{r.high_80}); "
                              f"nearest Safia-trained point {r.nearest_known_branch_km} km").add_to(ft)
    ft.add_to(m)
    cm.add_to(m)
    folium.LayerControl().add_to(m)
    return m


if __name__ == "__main__":
    sm = load()
    grid = tashkent_grid(sm)
    grid.to_parquet(ROOT / "data/processed/tashkent_grid.parquet")
    b = pd.read_csv(ROOT / "data/processed/branches.csv")
    build_map(grid, b, sm.fz.safia_net).save(ROOT / "reports/map.html")
    print(len(grid), "grid cells;", grid.sales_index.describe().round(1).to_dict())
    print(grid.nlargest(10, "sales_index")[["lat", "lon", "sales_index", "low_80", "high_80"]].to_string(index=False))
