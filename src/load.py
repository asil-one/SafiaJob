"""Load the Safia sample workbook into one tidy branch table."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data/raw/SampleSales_01-10_Sep_2026.xlsx"
OUT = ROOT / "data/processed/branches.csv"


def load_branches(path=XLSX) -> pd.DataFrame:
    sales = pd.read_excel(path, sheet_name="Sales index by day", header=3)
    sales = sales[pd.to_numeric(sales["Rank"], errors="coerce").notna()].copy()
    orders = pd.read_excel(path, sheet_name="Orders index by day", header=3)
    orders = orders[pd.to_numeric(orders.iloc[:, 0], errors="coerce").notna()].copy()

    day_cols = [c for c in sales.columns if isinstance(c, (float, int)) or str(c).endswith(".09")]
    df = sales.rename(columns={"Code": "code", "Branch": "name", "Format": "format",
                               "City (approx.)": "city", "Latitude": "lat", "Longitude": "lon",
                               "Avg per day": "sales", "Avg check index (top = 100)": "avg_check"})
    df["sales_cv"] = sales[day_cols].std(axis=1) / sales[day_cols].mean(axis=1)

    o_day = [c for c in orders.columns if isinstance(c, (float, int)) or str(c).endswith(".09")]
    orders["orders"] = orders[o_day].mean(axis=1)
    df = df.merge(orders[["Code", "orders"]].rename(columns={"Code": "code"}), on="code", how="left")

    df["city_clean"] = df["city"].str.replace(r"^near (\w+).*", r"\1", regex=True)
    df["is_tashkent"] = (df["city"] == "Tashkent").astype(int)
    df["is_mini"] = (df["format"] == "Safia mini").astype(int)
    cols = ["code", "name", "format", "city", "city_clean", "lat", "lon", "sales", "orders",
            "avg_check", "sales_cv", "is_tashkent", "is_mini"]
    return df[cols].astype({"lat": float, "lon": float, "sales": float}).reset_index(drop=True)


if __name__ == "__main__":
    df = load_branches()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.shape)
    print(df.head())
    print(df.describe().T)
