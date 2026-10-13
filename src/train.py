"""Compare models with spatial cross-validation, then fit the final model.

CV: branches are clustered by location (KMeans on lat/lon) and whole clusters are held out,
repeated with several seeds. This mimics "a branch the model has never seen, in an area
it may not have seen", which is how Safia evaluates. Random K-fold would be optimistic.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
META_COLS = ["code", "name", "format", "city", "city_clean", "lat", "lon", "sales", "orders",
             "avg_check", "sales_cv", "is_mini"]

# compact, interpretable set chosen from the EDA and the location-factor framework in the brief
COMPACT = ["n_supermarket_500", "n_shop_other_300", "n_fast_food_300", "n_bank_500", "n_cafe_300",
           "n_chain_300", "n_competitor_500", "n_building_1000", "pop_2000", "pop_10000",
           "n_safia_1000", "n_safia_2000", "d_safia", "d_metro", "d_mall", "d_major_road",
           "d_bus_stop", "is_tashkent"]


def models():
    return {
        "ridge_all": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))),
        "ridge_compact": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))),
        "rf": lambda: RandomForestRegressor(500, min_samples_leaf=5, max_features=0.33, random_state=0, n_jobs=-1),
        "extratrees": lambda: ExtraTreesRegressor(500, min_samples_leaf=5, max_features=0.33, random_state=0, n_jobs=-1),
        "gbm": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=7,
                                                     min_samples_leaf=10, l2_regularization=5, random_state=0),
        "rf_compact": lambda: RandomForestRegressor(500, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=-1),
    }


def feature_set(name, all_feats):
    return COMPACT if "compact" in name else all_feats


def spatial_cv(df, feats_all, n_clusters=12, seeds=(0, 1, 2, 3, 4)):
    y = np.log(df.sales.values)
    results, oof_store = {}, {}
    for name, make in {"median": None, **models()}.items():
        feats = feature_set(name, feats_all)
        maes, rhos, oofs = [], [], []
        for s in seeds:
            groups = KMeans(n_clusters, n_init=10, random_state=s).fit_predict(df[["lat", "lon"]] * [111, 84])
            oof = np.zeros(len(df))
            for tr, te in GroupKFold(6).split(df, y, groups):
                if make is None:
                    oof[te] = np.median(y[tr])
                else:
                    m = make().fit(df.iloc[tr][feats], y[tr])
                    oof[te] = m.predict(df.iloc[te][feats])
            pred = np.exp(oof)
            maes.append(np.mean(np.abs(pred - df.sales)))
            rhos.append(spearmanr(pred, df.sales).statistic if make else 0.0)
            oofs.append(pred)
        results[name] = {"MAE": np.mean(maes), "MAE_sd": np.std(maes), "spearman": np.mean(rhos)}
        oof_store[name] = np.mean(oofs, axis=0)
    return pd.DataFrame(results).T.sort_values("MAE"), oof_store


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "data/processed/features.parquet")
    feats = [c for c in df.columns if c not in META_COLS]
    res, oof = spatial_cv(df, feats)
    print(res.round(3).to_string())
    res.to_csv(ROOT / "reports/cv_results.csv")
    pd.DataFrame({"code": df.code, "sales": df.sales, **{k: v for k, v in oof.items()}}).to_csv(
        ROOT / "reports/oof_predictions.csv", index=False)
