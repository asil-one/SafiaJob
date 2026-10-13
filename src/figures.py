"""Static figures for the report (reports/figures/*.png)."""
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import NICE  # noqa: E402
from train import COMPACT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures"
BLUE, GRAY, INK, INK2, SURF = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#d6d5cf",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True})


def model_comparison():
    r = pd.read_csv(ROOT / "reports/cv_results.csv", index_col=0)
    oof = pd.read_csv(ROOT / "reports/oof_predictions.csv")
    ens = np.exp(np.log(oof[["rf_compact", "extratrees"]]).mean(axis=1))
    r.loc["rf+extratrees (final)"] = [np.mean(np.abs(ens - oof.sales)), np.nan, spearmanr(ens, oof.sales).statistic]
    r = r.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [GRAY if i == "median" else BLUE for i in r.index]
    ax.barh(r.index, r.MAE, color=colors, height=0.6)
    for y, (v, s) in enumerate(zip(r.MAE, r.spearman)):
        ax.text(v + 0.08, y, f"{v:.2f}   ρ={s:.2f}", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, r.MAE.max() * 1.3)
    ax.set_xlabel("Mean absolute error, sales index (lower is better) — spatial CV")
    ax.set_title("Model comparison vs. 'always predict the median' (grey)", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=160)


def pred_vs_actual():
    oof = pd.read_csv(ROOT / "reports/oof_predictions.csv")
    p = np.exp(np.log(oof[["rf_compact", "extratrees"]]).mean(axis=1))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([1, 90], [1, 90], color=GRAY, lw=1, ls="--")
    ax.scatter(oof.sales, p, s=28, color=BLUE, alpha=0.75, edgecolor=SURF, linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [2, 5, 10, 20, 50, 90]
    ax.set_xticks(ticks, ticks); ax.set_yticks(ticks, ticks)
    ax.set_xlabel("Actual avg daily sales index"); ax.set_ylabel("Predicted (out-of-fold)")
    ax.set_title(f"Held-out predictions — Spearman ρ={spearmanr(p, oof.sales).statistic:.2f}", loc="left", fontsize=11)
    top = oof.assign(p=p).nlargest(1, "sales").iloc[0]
    ax.annotate("Мега (mall)", (top.sales, top.p), xytext=(-60, 40), textcoords="offset points", color=INK2,
                fontsize=9, arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8})
    fig.tight_layout()
    fig.savefig(OUT / "pred_vs_actual.png", dpi=160)


def importance():
    blob = joblib.load(ROOT / "models/model.joblib")
    imp = pd.Series(np.mean([m.feature_importances_ for m in blob["models"]], axis=0), index=COMPACT)
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([NICE[c] for c in imp.index], imp.values, color=BLUE, height=0.6)
    ax.set_xlabel("Mean impurity importance (RF + ExtraTrees)")
    ax.set_title("What the model relies on", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=160)


def cannibalization():
    df = pd.read_parquet(ROOT / "data/processed/features.parquet")
    d = np.expm1(df.d_safia)
    bins = pd.cut(d, [0, 400, 800, 1500, 3000, 1e9], labels=["<400 m", "400–800 m", "0.8–1.5 km", "1.5–3 km", ">3 km"])
    g = df.groupby(bins, observed=True).sales.agg(["median", "count"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(g.index.astype(str), g["median"], color=BLUE, width=0.6)
    for x, (m, n) in enumerate(g.values):
        ax.text(x, m + 0.4, f"{m:.1f}\nn={int(n)}", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_ylim(0, g["median"].max() * 1.35)
    ax.set_ylabel("Median sales index")
    ax.set_xlabel("Distance to the nearest other Safia (file + 2GIS + OSM network)")
    ax.set_title("Cannibalization: isolated branches sell more", loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "cannibalization.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model_comparison(); pred_vs_actual(); importance(); cannibalization()
    print(sorted(p.name for p in OUT.glob("*.png")))
