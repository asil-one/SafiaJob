"""Turn any (lat, lon) into a feature vector describing its surroundings.

The same `Featurizer.transform` is used for training branches and for new locations,
so there is no train/serve skew. Everything is computed from cached local data:
  - OSM POIs (src/osm.py), Kontur population H3 grid, the Safia branch list.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
R_EARTH = 6_371_000.0
SELF_RADIUS = 150  # m: a Safia point this close is treated as the site itself (sources disagree by up to ~100 m)
DEDUPE_RADIUS = 120  # m: points from different sources closer than this are the same branch

CHAINS = ["evos", "oqtepa", "kfc", "max way", "maxway", "les ailes", "bellissimo", "feed up",
          "wendy", "yaponamama", "street 77", "bon!", "starbucks", "burger king", "loook"]
COMPETITOR_BAKERY = ["bon!", "бабушки майи", "дока хлеб", "paul", "chocolate", "cake", "торт", "кондитер",
                     "pâtisserie", "patisserie", "bakery", "пекарня"]
TASHKENT_CENTER = (41.3111, 69.2797)

COUNT_CATS = {  # category -> radii in metres
    "cafe": [300, 1000], "restaurant": [300, 1000], "fast_food": [300, 1000], "bakery": [500, 1000],
    "supermarket": [500, 1000], "shop_other": [300, 1000], "school": [1000], "kindergarten": [1000],
    "university": [1000], "hospital": [1000], "office": [500, 1000], "bank": [500], "bus_stop": [500],
    "parking": [300], "apartments": [500], "building": [300, 1000], "hotel": [1000], "pharmacy": [500],
}
DIST_CATS = ["metro", "mall", "marketplace", "university", "hospital", "major_road", "secondary_road",
             "bus_stop", "rail_station", "park"]


def _rad(lat, lon):
    return np.radians(np.column_stack([lat, lon]))


def _dedupe(pts, radius):
    """Greedy: keep a point only if no kept point lies within `radius` metres (sources listed in priority order)."""
    kept = []
    for la, lo in pts[["lat", "lon"]].values:
        if not kept or np.min(np.hypot((np.array(kept)[:, 0] - la) * 111_000,
                                       (np.array(kept)[:, 1] - lo) * 83_500)) > radius:
            kept.append((la, lo))
    return pd.DataFrame(kept, columns=["lat", "lon"])


class Featurizer:
    def __init__(self, branches: pd.DataFrame | None = None):
        pois = pd.read_parquet(ROOT / "data/external/osm_pois.parquet")
        pop = pd.read_parquet(ROOT / "data/external/population_h3.parquet")
        self.trees = {c: BallTree(_rad(g.lat, g.lon), metric="haversine") for c, g in pois.groupby("cat")}

        food = pois[pois.cat.isin(["cafe", "bakery", "fast_food", "restaurant"])].drop_duplicates("osm_id")
        label = food.brand.fillna(food.name).fillna("").str.lower()
        chain = food[label.apply(lambda s: any(k in s for k in CHAINS))]
        comp = food[label.apply(lambda s: "safia" not in s and "сафия" not in s
                                and any(k in s for k in COMPETITOR_BAKERY))
                    | ((food.cat == "bakery") & ~label.str.contains("safia"))]
        self.trees["chain"] = BallTree(_rad(chain.lat, chain.lon), metric="haversine")
        self.trees["competitor"] = BallTree(_rad(comp.lat, comp.lon), metric="haversine")

        # Safia network = file branches + 2GIS + OSM-tagged Safia (the file misses ~130 branches)
        sources = [] if branches is None else [branches[["lat", "lon"]]]
        dgis_net = ROOT / "data/external/dgis_safia.csv"
        if dgis_net.exists():
            sources.append(pd.read_csv(dgis_net)[["lat", "lon"]])
        sources.append(food[label.str.contains("safia|safiya|сафия")][["lat", "lon"]])
        net = _dedupe(pd.concat(sources, ignore_index=True), DEDUPE_RADIUS)
        self.safia_net = net
        self.safia_tree = BallTree(_rad(net.lat, net.lon), metric="haversine")

        self.pop_tree = BallTree(_rad(pop.lat, pop.lon), metric="haversine")
        self.pop_vals = pop.population.values

    def _count(self, tree, X, r):
        return tree.query_radius(X, r / R_EARTH, count_only=True)

    def _dist(self, tree, X):
        d, _ = tree.query(X, k=1)
        return d[:, 0] * R_EARTH

    def transform(self, lat, lon) -> pd.DataFrame:
        lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
        X = _rad(lat, lon)
        f = {}
        for cat, radii in COUNT_CATS.items():
            for r in radii:
                f[f"n_{cat}_{r}"] = np.log1p(self._count(self.trees[cat], X, r))
        for cat in DIST_CATS:
            f[f"d_{cat}"] = np.log1p(self._dist(self.trees[cat], X))
        for r in [300, 1000]:
            f[f"n_chain_{r}"] = np.log1p(self._count(self.trees["chain"], X, r))
        f["n_competitor_500"] = np.log1p(self._count(self.trees["competitor"], X, 500))
        f["d_competitor"] = np.log1p(self._dist(self.trees["competitor"], X))
        f["d_chain"] = np.log1p(self._dist(self.trees["chain"], X))

        # population within radius (sum of H3 res-8 cell centroids)
        for r in [500, 1000, 2000, 10000]:
            idx = self.pop_tree.query_radius(X, r / R_EARTH)
            f[f"pop_{r}"] = np.log1p([self.pop_vals[i].sum() for i in idx])

        # Safia network: neighbours excluding the site itself
        for r in [500, 1000, 2000]:
            f[f"n_safia_{r}"] = (self._count(self.safia_tree, X, r)
                                 - self._count(self.safia_tree, X, SELF_RADIUS))
        d, _ = self.safia_tree.query(X, k=min(4, self.safia_tree.data.shape[0]))
        d = d * R_EARTH
        nn = np.where(d[:, 0] < SELF_RADIUS, d[:, 1], d[:, 0])
        f["d_safia"] = np.log1p(nn)

        dc = np.hypot((lat - TASHKENT_CENTER[0]) * 111.0, (lon - TASHKENT_CENTER[1]) * 83.5)
        f["is_tashkent"] = (dc < 18).astype(float)
        f["d_tashkent_center"] = np.log1p(dc)
        return pd.DataFrame(f)


if __name__ == "__main__":
    b = pd.read_csv(ROOT / "data/processed/branches.csv")
    fz = Featurizer(b)
    F = fz.transform(b.lat.values, b.lon.values)
    out = pd.concat([b.drop(columns=["is_tashkent"]), F], axis=1)
    out.to_parquet(ROOT / "data/processed/features.parquet")
    from scipy.stats import spearmanr
    rho = {c: spearmanr(F[c], b.sales).statistic for c in F.columns}
    print(pd.Series(rho).sort_values().to_string())
