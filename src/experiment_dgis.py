"""Experiment: do 2GIS business counts add signal on top of the OSM features?

2GIS coverage is missing in most regional cities (all counts = 0), so those rows are set to NaN
instead of being read as "nothing nearby". Same spatial CV as train.py.
Result (Oct 2026): +g_beauty_500 MAE 8.23 -> 8.20, all six 8.26: within noise, not adopted.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import COMPACT  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def cv(df, y, feats, seeds=range(5)):
    maes, rhos = [], []
    for s in seeds:
        grp = KMeans(12, n_init=10, random_state=s).fit_predict(df[["lat", "lon"]] * [111, 84])
        oof = np.zeros(len(df))
        for tr, te in GroupKFold(6).split(df, y, grp):
            ms = [RandomForestRegressor(500, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=-1),
                  ExtraTreesRegressor(500, min_samples_leaf=5, max_features=0.33, random_state=0, n_jobs=-1)]
            oof[te] = np.mean([m.fit(df.iloc[tr][feats], y[tr]).predict(df.iloc[te][feats]) for m in ms], axis=0)
        p = np.exp(oof)
        maes.append(np.mean(np.abs(p - df.sales)))
        rhos.append(spearmanr(p, df.sales).statistic)
    return np.mean(maes), np.mean(rhos)


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "data/processed/features.parquet")
    g = pd.read_csv(ROOT / "data/processed/dgis_features.csv")
    G = [c for c in g if c.startswith("g_")]
    covered = g[G].sum(axis=1) > 0
    for c in G:
        df[c] = np.where(covered, g[c], np.nan)
    y = np.log(df.sales.values)
    print(f"2GIS covered: {covered.sum()}/{len(df)} branches")
    rows = {}
    for name, f in [("OSM + Safia network", COMPACT), ("+ beauty salons", COMPACT + ["g_beauty_500"]),
                    ("+ beauty, cafe, fast food", COMPACT + ["g_beauty_500", "g_cafe_500", "g_fastfood_500"]),
                    ("+ all six 2GIS counts", COMPACT + G)]:
        rows[name] = cv(df, y, f)
        print(name, np.round(rows[name], 3))
    pd.DataFrame(rows, index=["MAE", "spearman"]).T.to_csv(ROOT / "reports/experiment_dgis.csv")
