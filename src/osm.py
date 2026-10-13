"""Extract OSM points of interest for Uzbekistan from the Geofabrik extract and cache as parquet.

Nodes keep their location, ways are reduced to the mean of their node locations.
Road ways (major/secondary) are stored as all of their nodes so we can measure distance to the road.
Download: https://download.geofabrik.de/asia/uzbekistan-latest.osm.pbf -> data/external/
"""
import re
from pathlib import Path

import numpy as np
import osmium
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PBF = ROOT / "data/external/uzbekistan.osm.pbf"
CACHE = ROOT / "data/external/osm_pois"  # folder of parquet parts (small files push reliably)

FOOD_SHOPS = {"bakery", "confectionery", "pastry", "chocolate"}
GROCERY = {"supermarket", "convenience"}


def categorize(t):
    """Return the list of categories a tag set belongs to."""
    a, s, h = t.get("amenity"), t.get("shop"), t.get("highway")
    cats = []
    if a == "cafe": cats.append("cafe")
    if a == "restaurant": cats.append("restaurant")
    if a == "fast_food": cats.append("fast_food")
    if s in FOOD_SHOPS: cats.append("bakery")
    if s in GROCERY: cats.append("supermarket")
    if s in {"mall", "department_store"}: cats.append("mall")
    if s and s not in FOOD_SHOPS | GROCERY | {"mall", "department_store"}: cats.append("shop_other")
    if a == "marketplace": cats.append("marketplace")
    if a == "school": cats.append("school")
    if a == "kindergarten": cats.append("kindergarten")
    if a in {"university", "college"}: cats.append("university")
    if a in {"hospital", "clinic"}: cats.append("hospital")
    if a == "pharmacy": cats.append("pharmacy")
    if a in {"bank", "atm"}: cats.append("bank")
    if "office" in t: cats.append("office")
    if h == "bus_stop" or t.get("public_transport") == "platform": cats.append("bus_stop")
    if t.get("station") == "subway" or t.get("subway") == "yes" and t.get("railway") == "station": cats.append("metro")
    if t.get("railway") == "station" and t.get("station") != "subway": cats.append("rail_station")
    if a == "parking": cats.append("parking")
    if a == "fuel": cats.append("fuel")
    if a == "place_of_worship": cats.append("worship")
    if t.get("leisure") == "park": cats.append("park")
    if t.get("tourism") == "hotel": cats.append("hotel")
    if t.get("building") == "apartments": cats.append("apartments")
    if t.get("building") and t.get("building") != "apartments": cats.append("building")
    return cats


class Handler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.rows, self.road_pts = [], []

    def _add(self, kind, oid, lat, lon, t):
        cats = categorize(t)
        if not cats:
            return
        name = t.get("name") or t.get("name:ru") or t.get("name:uz")
        for c in cats:
            self.rows.append((c, f"{kind}/{oid}", lat, lon, name, t.get("brand")))

    def node(self, n):
        if len(n.tags) and n.location.valid():
            self._add("n", n.id, n.location.lat, n.location.lon, dict(n.tags))

    def way(self, w):
        t = dict(w.tags)
        hw = t.get("highway")
        pts = [(nd.lat, nd.lon) for nd in w.nodes if nd.location.valid()]
        if not pts:
            return
        if hw in {"motorway", "trunk", "primary", "secondary", "tertiary"}:
            cls = "major_road" if hw in {"motorway", "trunk", "primary"} else "secondary_road"
            self.road_pts += [(cls, f"w/{w.id}", la, lo, t.get("name"), None) for la, lo in pts]
        if t:
            lat = sum(p[0] for p in pts) / len(pts)
            lon = sum(p[1] for p in pts) / len(pts)
            self._add("w", w.id, lat, lon, t)


def load_pois(refresh=False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    h = Handler()
    h.apply_file(str(PBF), locations=True, idx="flex_mem")
    cols = ["cat", "osm_id", "lat", "lon", "name", "brand"]
    pois = pd.DataFrame(h.rows + h.road_pts, columns=cols)
    food = pois.cat.isin(["cafe", "bakery", "fast_food", "restaurant"])
    pois.loc[~food, ["name", "brand"]] = None  # names are only used for food places
    CACHE.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(np.array_split(np.arange(len(pois)), 6)):
        pois.iloc[idx].to_parquet(CACHE / f"part-{i}.parquet", compression="zstd", compression_level=19, index=False)
    return pois


if __name__ == "__main__":
    p = load_pois(refresh=True)
    print(p["cat"].value_counts().to_string())
