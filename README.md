# The Next Safia: predicting a location's sales index

Give it a location and it returns the expected **sales index (1–100)** of a Safia branch there, on the same scale as `SampleSales_01-10_Sep_2026.xlsx`. It also returns an 80% range, a confidence flag and the factors that drive the number.

The research write-up is in **[REPORT.md](REPORT.md)**.

## Quick start

```bash
# 1. install (needs uv: https://docs.astral.sh/uv/)
uv sync

# 2. one location
uv run python src/predict.py --lat 41.3650 --lon 69.2880

# 3. many locations (CSV with lat, lon columns; other columns are kept)
uv run python src/predict.py --csv data/example_locations.csv --out predictions.csv

# 4. interactive map: click anywhere to get a prediction plus a heatmap of Tashkent
uv run streamlit run app/app.py
```

Example output:
```
Predicted sales index: 22.9  (80% range 12.7 - 49.3)
Confidence: normal; nearest known branch 0.37 km
Main drivers vs a typical branch:
  supermarkets/convenience within 500 m +2.7
  ...
```

`reports/map.html` is a static map you can open in any browser, with no server needed. It shows the predicted-potential heatmap, all known Safia branches and the training branches coloured by actual sales.

**Prediction makes no paid API calls.** All data sources are cached in `data/external/`.

## Rebuild everything from scratch

```bash
uv run python src/load.py          # xlsx -> data/processed/branches.csv
# OSM extract: https://download.geofabrik.de/asia/uzbekistan-latest.osm.pbf -> data/external/
uv run python src/osm.py           # -> data/external/osm_pois.parquet
# population: Kontur H3 grid (HDX "kontur-population-uzbekistan") -> data/external/population_h3.parquet
uv run python src/dgis.py          # 2GIS Safia network + business counts (needs DGIS_API_KEY in .env)
uv run python src/features.py      # -> data/processed/features.parquet
uv run python src/train.py         # spatial CV model comparison -> reports/cv_results.csv
uv run python src/predict.py --train
uv run python src/make_maps.py     # Tashkent grid + reports/map.html
uv run python src/figures.py       # reports/figures/*.png
uv run python src/experiment_dgis.py
```

## Layout

| Path | What |
|---|---|
| `src/load.py` | Parse the workbook into one tidy table per branch |
| `src/osm.py` | Extract OSM POIs (26 categories) from the Geofabrik file |
| `src/dgis.py` | 2GIS Places API: full Safia network and business counts (cached) |
| `src/features.py` | `Featurizer.transform(lat, lon)`, the single feature code path for training and prediction |
| `src/train.py` | Model comparison with spatially grouped CV |
| `src/predict.py` | Final model, CLI, intervals, explanations |
| `src/make_maps.py`, `src/figures.py` | Map and figures |
| `src/experiment_dgis.py` | Experiment: do 2GIS business counts help? (not adopted) |
| `app/app.py` | Streamlit app |
| `reports/` | CV results, out-of-fold predictions, figures, map |
