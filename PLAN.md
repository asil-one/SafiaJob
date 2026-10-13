# The Next Safia: Plan

## 1. The task

Build a system that takes a proposed location (lat/lon) and returns an **expected sales index from 1 to 100**.
Safia will score it on **real branches that are not in our file**, and may also give us a new location to assess.

What we have to hand in:
1. Working model and code, with a simple way to enter a location and get a prediction.
2. A short write-up covering the research process, the factors used, data sources, assumptions and validation.
3. Extras: maps, experiments, uncertainty estimates, ideas that failed, insights. The brief says they care most about *how we think*.

## 2. What the data contains

`SampleSales_01-10_Sep_2026.xlsx` has 4 sheets:

| Sheet | Contents |
|---|---|
| Sales index by day | 127 branches: code, name, format, city, lat/lon, sales index for each day 1–10 Sep, avg per day, avg check index |
| Orders index by day | Same layout, but for orders |
| Daily detail | Long format with one row per branch and day (1,270 rows): sales index and orders index |
| About | Method notes (index = highest single-day sales of any branch = 100; source is iiko) |

What we see so far:
- **The target** is probably `Avg per day` (10-day mean sales index). It is right-skewed: min 1.6, median 17.8, mean 20.4, p90 35.5, max 86.5 (W035 "Мега", likely the Mega Planet mall).
- **The data is concentrated in Tashkent**: 102 of 127 branches. The rest are spread thinly over about 19 regional cities, often with 1–4 branches each. Predicting regional branches will be the hard part.
- **The sample is very small (127)**. We need simple, regularized models, careful cross-validation, and good features rather than a large model.
- **Format**: 125 are "Safia" and 2 are "Safia mini".
- **The network is incomplete**. According to "About", 68 branches have no location, 14 have a wrong location, 14 have partial sales and 17 have no sales. Safia's hidden test branches probably come from these groups. This means our "nearby Safia branches" (cannibalization) features undercount real neighbours, and we should say so in the write-up.
- **Calendar effects**: 1 Sep is Independence Day and 2 Sep is the first school day. We'll check whether these days look unusual and say why we average over all 10 days.
- Branch names carry hints (mall, bazaar, hospital, metro and so on). These are useful for research, but a new location has no name, so a name can only be an optional input.

## 3. Approach

### Phase A: Understand the data (days 1–2)
- Clean and reshape the data. Check the coordinates on a map (look for duplicates, points that fall in the wrong city, and the "near X km" cases).
- EDA: distribution of the target, daily stability (CV for each branch), how sales split into orders × avg check, Tashkent vs regions, and a spatial map of sales. Check whether high-sales branches cluster together (Moran's I).
- Decide on the target: `log(avg daily sales index)`, transformed back and clipped to [1, 100].

### Phase B: Feature engineering, turning a location into evidence (days 2–6)
For every point, compute counts and distances within buffers of 250 m, 500 m, 1 km and 2 km:

| Group | Features | Source |
|---|---|---|
| **Competition** | Bakeries, cafes, confectioneries, coffee shops, and named competitors nearby, plus the distance to the nearest one | OSM (Overpass); possibly 2GIS or Yandex, Google Places |
| **People & activity** | Population density, POI density, offices, universities, schools, hospitals, malls, bazaars, supermarkets | WorldPop / Kontur population (H3) / GHSL, OSM |
| **Accessibility** | Distance to the nearest metro station or bus stop, road density, distance to a major road, intersection density, parking | OSM (metro, highway=primary/trunk), OSMnx |
| **Economic proxy** | Night-time lights (VIIRS), building density and height | NASA VIIRS, OSM buildings / Microsoft or Google Open Buildings |
| **Safia context** | Distance to the nearest Safia branch, number of Safia branches within 1 or 2 km (cannibalization), distance to the city centre, city size (population), Tashkent vs region, format | Our file plus city population figures |
| **Spatial lag** (test carefully) | Weighted average sales of the k nearest *other* branches | Our file (exclude the branch itself in CV) |

The work is stored as one reusable function `featurize(lat, lon) -> vector`, with the OSM and raster data cached locally. That way training and prediction run exactly the same code.

### Phase C: Modelling and validation (days 6–10)
- **Baselines**: global median, city median, kNN on coordinates. Every other model has to beat these.
- **Models**: Ridge or ElasticNet on a few interpretable features, LightGBM or CatBoost with strong regularization (shallow trees, monotone constraints where they make sense), and possibly a GP or kriging on the residuals.
- **Validation that matches the real test**:
  - **Spatial / grouped K-fold** (cluster the points, e.g. KMeans on coordinates, and hold out a whole cluster) instead of random CV, because random CV leaks from neighbours.
  - **Leave-one-city-out** for the regional branches.
  - Metrics: MAE and RMSE on the 1–100 scale, Spearman rank correlation (useful for ranking candidate sites), and error broken down by Tashkent vs regions.
- **Uncertainty**: quantile models or conformal prediction intervals, so the output is e.g. "predicted 24 (80% interval 15–36)". Also add a flag when a location is outside the training distribution (a region with no branches).
- **Interpretability**: SHAP values for each prediction, so a site report can say *why*, e.g. "near metro +6, 3 Safia branches within 1 km −4".
- Keep a log of experiments, including the ones that failed (this is explicitly requested).

### Phase D: Product and delivery (days 10–13)
- **CLI**: `python predict.py --lat 41.31 --lon 69.28` prints the index, the interval and the main factors.
- **Small web app** (Streamlit or Folium): click on a map to get a prediction and an explanation, plus a heatmap of predicted potential across Tashkent ("where should the next Safia go?"). This is the extra skill that will be most visible to the reviewers.
- **Batch mode**: a CSV of locations in, a CSV of predictions out. This is how they will evaluate the hidden branches.
- `README.md` with setup steps (`requirements.txt` or `uv`) and examples.

### Phase E: Write-up (days 12–14)
A 3–5 page report or notebook covering the problem framing, EDA findings, the factors and why they were chosen, data sources, assumptions and limitations (10 days of data in September, a partial network, no info on store size or rent), validation results compared with the baselines, uncertainty, failed ideas, and next steps (more months of data, footfall data, store attributes, A/B tests of the site-selection process).

## 4. Proposed folder structure
```
sofia/
  data/raw/            # the xlsx, the case PDF
  data/external/       # cached OSM, population, VIIRS
  data/processed/      # features.parquet
  notebooks/           # 01_eda, 02_features, 03_modeling
  src/                 # load.py, features.py, train.py, predict.py
  app/                 # streamlit map app
  models/              # trained model + conformal calibration
  reports/             # write-up, figures, maps
  README.md
```

## 5. Risks and decisions to settle
- **Scale of the hidden branches**: we assume their index uses the same scale (max day in *our* file = 100), so predictions are in the same units. Confirm this with Safia if we can.
- **Regional branches** have few training examples. We'll rely on city-level features (population, city size) and wide uncertainty intervals.
- **OSM coverage in Uzbekistan** may be patchy. Compare it with 2GIS or Yandex for a sample of points. If an API key or scraping is needed, check the terms first.
- **Deadline**: 14 days from when the email arrived. We need the exact date to fix the schedule.
