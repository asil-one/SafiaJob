"""Predict the 1-100 sales index for any location.

Usage:
  uv run python src/predict.py --lat 41.311 --lon 69.279
  uv run python src/predict.py --csv locations.csv --out predictions.csv   # columns: lat, lon (+ anything)
  uv run python src/predict.py --train      # refit the final model + calibrate intervals
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import Featurizer  # noqa: E402
from train import COMPACT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/model.joblib"
NICE = {
    "n_supermarket_500": "supermarkets/convenience within 500 m", "n_shop_other_300": "shops within 300 m",
    "n_fast_food_300": "fast-food outlets within 300 m", "n_bank_500": "banks/ATMs within 500 m",
    "n_cafe_300": "cafés within 300 m", "n_chain_300": "big food chains within 300 m",
    "n_competitor_500": "bakeries/confectioners within 500 m", "n_building_1000": "building density (1 km)",
    "pop_2000": "population within 2 km", "pop_10000": "population within 10 km (city size)",
    "n_safia_1000": "other Safia within 1 km", "n_safia_2000": "other Safia within 2 km",
    "d_safia": "distance to nearest Safia", "d_metro": "distance to metro", "d_mall": "distance to mall",
    "d_major_road": "distance to major road", "d_bus_stop": "distance to bus stop", "is_tashkent": "in Tashkent",
}


class SalesModel:
    def __init__(self, models, featurizer, train_X, train_latlon, resid_q):
        self.models, self.fz, self.train_X = models, featurizer, train_X
        self.train_tree = BallTree(np.radians(train_latlon), metric="haversine")
        self.resid_q = resid_q  # quantiles of out-of-fold log residuals -> calibrated interval

    def _raw(self, X):
        return np.mean([m.predict(X[COMPACT]) for m in self.models], axis=0)  # log scale

    def predict(self, lat, lon, explain=True):
        F = self.fz.transform(lat, lon)
        logp = self._raw(F)
        out = pd.DataFrame({"lat": np.atleast_1d(lat), "lon": np.atleast_1d(lon)})
        out["sales_index"] = np.clip(np.exp(logp), 1, 100).round(1)
        out["low_80"] = np.clip(np.exp(logp + self.resid_q[0]), 1, 100).round(1)
        out["high_80"] = np.clip(np.exp(logp + self.resid_q[1]), 1, 100).round(1)
        d_km = self.train_tree.query(np.radians(out[["lat", "lon"]].values), k=1)[0][:, 0] * 6371
        out["nearest_known_branch_km"] = d_km.round(2)
        out["confidence"] = np.where(d_km > 30, "low (area not covered by training data)",
                                     np.where(d_km > 5, "medium", "normal"))
        if explain:
            out["drivers"] = [self._explain(F.iloc[[i]], logp[i]) for i in range(len(F))]
        return out, F

    def _explain(self, row, base):
        """Local what-if: replace each feature with the network median, see how the prediction moves."""
        med = self.train_X[COMPACT].median()
        eff = {}
        for c in COMPACT:
            alt = row.copy()
            alt[c] = med[c]
            eff[c] = np.exp(base) - np.exp(self._raw(alt)[0])
        top = sorted(eff.items(), key=lambda kv: -abs(kv[1]))[:5]
        return "; ".join(f"{NICE[c]} {v:+.1f}" for c, v in top if abs(v) >= 0.3)


def make_models():
    return [RandomForestRegressor(500, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=-1),
            ExtraTreesRegressor(500, min_samples_leaf=5, max_features=0.33, random_state=0, n_jobs=-1)]


def train():
    df = pd.read_parquet(ROOT / "data/processed/features.parquet")
    y = np.log(df.sales.values)
    oof = pd.read_csv(ROOT / "reports/oof_predictions.csv")
    ens = np.exp(np.log(oof[["rf_compact", "extratrees"]]).mean(axis=1))
    resid = np.log(oof.sales) - np.log(ens)
    q = tuple(np.quantile(resid, [0.10, 0.90]))
    models = [m.fit(df[COMPACT], y) for m in make_models()]
    MODEL.parent.mkdir(exist_ok=True)
    # store plain data only (fitted estimators + calibration); feature sources are rebuilt on load
    joblib.dump({"models": models, "resid_q": q, "train_X": df[COMPACT], "latlon": df[["lat", "lon"]].values},
                MODEL, compress=3)
    print(f"saved {MODEL}; 80% interval multipliers x{np.exp(q[0]):.2f} .. x{np.exp(q[1]):.2f}")
    return _build(joblib.load(MODEL))


def _build(blob):
    branches = pd.read_csv(ROOT / "data/processed/branches.csv")
    return SalesModel(blob["models"], Featurizer(branches), blob["train_X"], blob["latlon"], blob["resid_q"])


def load():
    return _build(joblib.load(MODEL)) if MODEL.exists() else train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--csv")
    ap.add_argument("--out")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    if a.train:
        train()
    sm = load()
    if a.csv:
        inp = pd.read_csv(a.csv)
        res, _ = sm.predict(inp.lat.values, inp.lon.values)
        res = pd.concat([inp.drop(columns=["lat", "lon"]), res], axis=1)
        res.to_csv(a.out or "predictions.csv", index=False)
        print(res.to_string(index=False))
    elif a.lat is not None:
        r, _ = sm.predict(a.lat, a.lon)
        r = r.iloc[0]
        print(f"Predicted sales index: {r.sales_index}  (80% range {r.low_80} - {r.high_80})")
        print(f"Confidence: {r.confidence}; nearest known branch {r.nearest_known_branch_km} km")
        print("Main drivers vs a typical branch:\n  " + r.drivers.replace("; ", "\n  "))
