"""2GIS Places API features: business counts around a point, and the full Safia network.

Every API response is cached in data/external/dgis_cache.json so repeated runs and
predictions for already-seen points cost no quota. Key is read from .env (DGIS_API_KEY).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/external/dgis_cache.json"
SAFIA_CSV = ROOT / "data/external/dgis_safia.csv"
URL = "https://catalog.api.2gis.com/3.0/items"

# feature name -> (search text, radius m). Text search over 2GIS's business directory.
# ALL_QUERIES was explored on 59 branches; QUERIES (the model's set) keeps the strongest + competition
# to fit the demo-key quota (~1000 requests/key).
ALL_QUERIES = {
    "g_cafe_500": ("кафе", 500),
    "g_coffee_500": ("кофейня", 500),
    "g_confectionery_1000": ("кондитерская", 1000),
    "g_bakery_1000": ("пекарня", 1000),
    "g_fastfood_500": ("фастфуд", 500),
    "g_grocery_500": ("продукты", 500),
    "g_mall_1000": ("торговый центр", 1000),
    "g_office_1000": ("бизнес-центр", 1000),
    "g_pharmacy_500": ("аптека", 500),
    "g_edu_1500": ("университет", 1500),
    "g_school_1000": ("школа", 1000),
    "g_clinic_1000": ("клиника", 1000),
    "g_beauty_500": ("салон красоты", 500),
    "g_clothes_500": ("одежда", 500),
}
QUERIES = {k: ALL_QUERIES[k] for k in ["g_clinic_1000", "g_beauty_500", "g_fastfood_500", "g_cafe_500",
                                      "g_office_1000", "g_confectionery_1000"]}
MAX_REQUESTS_PER_RUN = 600


def _key():
    """Newest key wins: NEW_DGIS_API_KEY, then DGIS_API_KEY (environment first, then .env)."""
    env = ROOT / ".env"
    vals = {}
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    for name in ("NEW_DGIS_API_KEY", "DGIS_API_KEY"):
        if os.environ.get(name) or vals.get(name):
            return os.environ.get(name) or vals[name]
    return None


class DGis:
    def __init__(self):
        self.key = _key()
        self.cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        self._dirty = 0
        self.requests = 0

    def save(self):
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(self.cache, ensure_ascii=False))
        self._dirty = 0

    def _get(self, params):
        if not self.key:
            raise RuntimeError("DGIS_API_KEY not set (.env)")
        if self.requests >= MAX_REQUESTS_PER_RUN:
            raise RuntimeError(f"request budget of {MAX_REQUESTS_PER_RUN} reached; rerun to continue")
        self.requests += 1
        for attempt in range(4):
            r = requests.get(URL, params={**params, "key": self.key}, timeout=30)
            d = r.json()
            code = d.get("meta", {}).get("code")
            if code in (200, 404):  # 404 = nothing found
                return d
            if code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"2GIS error {code}: {d.get('meta', {}).get('error')}")
        raise RuntimeError("2GIS retries exhausted")

    def count(self, lat, lon, q, radius):
        ck = f"{lat:.5f},{lon:.5f}|{q}|{radius}"
        if ck not in self.cache:
            d = self._get({"q": q, "point": f"{lon},{lat}", "radius": radius, "page_size": 1, "type": "branch"})
            self.cache[ck] = d.get("result", {}).get("total", 0)
            self._dirty += 1
            if self._dirty >= 25:
                self.save()
        return self.cache[ck]

    def features(self, lat, lon):
        lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
        rows = [{name: np.log1p(self.count(la, lo, q, r)) for name, (q, r) in QUERIES.items()}
                for la, lo in zip(lat, lon)]
        self.save()
        return pd.DataFrame(rows)

    def safia_network(self, centers):
        """All 2GIS 'Safia' branches within 40 km of each centre (lat, lon), deduped."""
        if SAFIA_CSV.exists():
            return pd.read_csv(SAFIA_CSV)
        found = {}

        def search(la, lo, r):
            # demo keys return at most 5 pages x 10; split the circle into 4 quadrants when it holds more
            params = {"q": "Safia", "point": f"{lo},{la}", "radius": int(r), "page_size": 10,
                      "type": "branch", "fields": "items.point,items.address"}
            d = self._get({**params, "page": 1})
            total = d.get("result", {}).get("total", 0)
            if total > 50 and r > 1500:
                dlat, dlon = r / 2 / 111_000, r / 2 / (111_000 * np.cos(np.radians(la)))
                for sy in (-1, 1):
                    for sx in (-1, 1):
                        search(la + sy * dlat, lo + sx * dlon, r * 0.75)
                return
            pages = [d] + [self._get({**params, "page": p}) for p in range(2, min(5, -(-total // 10)) + 1)]
            for pd_ in pages:
                for it in pd_.get("result", {}).get("items", []):
                    if "safia" in it.get("name", "").lower() and it.get("point"):
                        found[it["id"]] = {"id": it["id"], "name": it["name"], "address": it.get("address_name"),
                                           "lat": it["point"]["lat"], "lon": it["point"]["lon"]}

        for la, lo in centers:
            search(la, lo, 40000)
        out = pd.DataFrame(found.values())
        out.to_csv(SAFIA_CSV, index=False)
        return out


if __name__ == "__main__":
    b = pd.read_csv(ROOT / "data/processed/branches.csv")
    g = DGis()
    # one search centre per city cluster (rounded to ~0.3 deg) covers all known branch areas
    centers = b[["lat", "lon"]].round(1).drop_duplicates().values
    centers = [c for i, c in enumerate(centers)
               if all(np.hypot(*(c - p) * [111, 84]) > 30 for p in centers[:i])]
    net = g.safia_network(centers)
    print(f"2GIS Safia branches: {len(net)} (searched {len(centers)} centres)")
    F = g.features(b.lat.values, b.lon.values)
    out = pd.concat([b[["code", "sales"]], F], axis=1)
    out.to_csv(ROOT / "data/processed/dgis_features.csv", index=False)
    from scipy.stats import spearmanr
    print(pd.Series({c: spearmanr(F[c], b.sales).statistic for c in F.columns}).sort_values().round(3).to_string())
