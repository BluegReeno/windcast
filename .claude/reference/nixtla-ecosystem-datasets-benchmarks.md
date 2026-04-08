# Nixtla Ecosystem: Datasets and Benchmarks

Research date: 2026-04-02
Sources: GitHub repos, official docs, source code inspection

---

## 1. datasetsforecast — Complete Dataset Catalog

Package: `pip install datasetsforecast`
GitHub: https://github.com/Nixtla/datasetsforecast
Docs: https://nixtlaverse.nixtla.io/datasetsforecast/

All datasets return DataFrames in long format: `(unique_id, ds, y)`.
Most `load()` methods return a tuple: `(Y_df, X_df, S_df)` where:
- `Y_df`: target time series
- `X_df`: exogenous / dynamic features (None if absent)
- `S_df`: static features or summing matrix (None if absent)

---

### 1.1 M3 Competition (`datasetsforecast.m3`)

| Group     | n_series | Freq    | Horizon | Seasonality |
|-----------|----------|---------|---------|-------------|
| Yearly    | 645      | Annual  | 6       | 1           |
| Quarterly | 756      | Q       | 8       | 4           |
| Monthly   | 1,428    | M       | 18      | 12          |
| Other     | 174      | Daily   | 8       | 1           |

- **Domain**: Mixed (finance, industry, macro, demographics, micro)
- **Exogenous**: None
- **Source**: Zenodo (Monash repo)
- **Usage**: `M3.load(directory, group)` → `(Y_df, None, None)`

---

### 1.2 M4 Competition (`datasetsforecast.m4`)

| Group     | n_series | Freq    | Horizon |
|-----------|----------|---------|---------|
| Yearly    | 23,000   | Annual  | 6       |
| Quarterly | 24,000   | Q       | 8       |
| Monthly   | 48,000   | Monthly | 18      |
| Weekly    | 359      | Weekly  | 13      |
| Daily     | 4,227    | Daily   | 14      |
| Hourly    | 414      | Hourly  | 48      |

- **Domain**: Mixed (macro, finance, micro, industry, demographics)
- **Exogenous**: None (univariate only)
- **Usage**: `M4.load(directory, group)` → `(Y_df, None, None)`
- **Notes**: Nixtla has Kaggle notebooks for M4 at https://www.kaggle.com/code/lemuz90/m4-competition

---

### 1.3 M5 Competition (`datasetsforecast.m5`)

- **Domain**: Retail demand (Walmart stores, US)
- **n_series**: 30,490 item-store combinations
- **Frequency**: Daily
- **Horizon**: 28 days
- **Exogenous features (rich)**:
  - `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2` — cultural/sporting/national events
  - `snap_CA`, `snap_TX`, `snap_WI` — SNAP food stamp program (binary per state)
  - `sell_price` — weekly item prices per store
- **Static features**: `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`
- **Usage**: `M5.load(directory)` → `(Y_df, X_df, S_df)`
- **Notes**: Most exogenous-rich dataset in the collection. Excellent for validating exog pipelines.

---

### 1.4 Long Horizon Benchmark (`datasetsforecast.long_horizon`)

The standard benchmark suite for transformer-era papers (Informer, Autoformer, PatchTST, etc.).
All datasets have horizons: `(96, 192, 336, 720)`.

| Dataset  | n_series | Freq  | Domain              | Exogenous |
|----------|----------|-------|---------------------|-----------|
| ETTh1    | 1        | H     | Electricity transformer (China) | Yes (7 covariates returned as X_df) |
| ETTh2    | 1        | H     | Electricity transformer (China) | Yes |
| ETTm1    | 7        | 15min | Electricity transformer | Yes |
| ETTm2    | 7        | 15min | Electricity transformer | Yes |
| ECL      | 321      | 15min→H | Electricity consumption (KWh, customers) | Yes |
| Exchange | 8        | D     | FX rates (8 countries vs USD) | No meaningful exog |
| TrafficL | 862      | H     | Road occupancy (CA Dept of Transportation) | Yes |
| ILI      | 7        | W     | Influenza-like illness (CDC) | No |
| Weather  | 21       | 10min | Meteorological measurements (Max Planck, Jena) | Yes |

**Critical note**: `LongHorizon.load()` returns `(Y_df, X_df, S_df)`. The `X_df` contains
other channels of the multivariate dataset as exogenous features. These are already normalized
with train mean/std. This is the approach used in transformer papers — each channel can serve
as exogenous input to other channels.

**Energy-relevant datasets**:
- ETTh1, ETTh2, ETTm1, ETTm2: Electricity transformer temperature + load variants. **Most comparable to our use case.**
- ECL: 321 electricity customers, 2012-2014. Large multivariate electricity dataset.
- TrafficL: Road sensor occupancy.

---

### 1.5 Long Horizon v2 (`datasetsforecast.long_horizon2`)

Same datasets as above but with the original train/val/test split used in transformer papers
(not the 70/10/20 normalized version). Applies `StandardScaler` from sklearn.

| Dataset  | n_series | n_time  | val_size | test_size |
|----------|----------|---------|----------|-----------|
| ETTh1    | 7        | 14,400  | 2,880    | 2,880     |
| ETTh2    | 7        | 14,400  | 2,880    | 2,880     |
| ETTm1    | 7        | 57,600  | 11,520   | 11,520    |
| ETTm2    | 7        | 57,600  | 11,520   | 11,520    |
| ECL      | 321      | 26,304  | 2,632    | 5,260     |
| TrafficL | 862      | ~17,544 | 1,756    | 3,508     |
| Weather  | 21       | ~52,696 | 5,270    | 10,539    |

---

### 1.6 Hierarchical Datasets (`datasetsforecast.hierarchical`)

For hierarchical / reconciled forecasting research.

| Dataset       | Freq | Horizon | n_levels | Domain       |
|---------------|------|---------|----------|--------------|
| Labour        | MS   | 8/12    | 4        | Australian employment |
| TourismLarge  | MS   | 12      | 8        | Australian tourism |
| TourismSmall  | Q    | 4       | 4        | Australian tourism |
| Traffic       | D    | 14      | 4        | Traffic |
| Wiki2         | D    | 14      | 5        | Wikipedia page views |

- **Exogenous**: None (hierarchical structure only, no external regressors)
- Returns `(Y_df, S_df, tags)` where `S_df` is the summing/aggregation matrix

---

### 1.7 Favorita (`datasetsforecast.favorita`)

Subsets of the Corporación Favorita Kaggle competition (Ecuadorian grocery retailer).

| Group        | Freq | Horizon | Seasonality | n_levels |
|--------------|------|---------|-------------|----------|
| Favorita200  | D    | 34      | 7           | 4 (Country → Store) |
| Favorita500  | D    | 34      | 7           | 4 |

- **Domain**: Retail demand (hierarchical)
- **Exogenous**: Blog example shows oil prices, holidays, promotions are available in the
  full Kaggle dataset but NOT bundled in the datasetsforecast package version
- **Notes**: Primarily used for hierarchical forecasting experiments in Nixtla papers

---

### 1.8 PHM2008 — Turbofan Engine Degradation (`datasetsforecast.phm2008`)

| Group  | n_train | n_test | n_sensors |
|--------|---------|--------|-----------|
| FD001  | 100     | 100    | 16        |
| FD002  | 260     | 259    | 16        |
| FD003  | 100     | 100    | 16        |
| FD004  | 249     | 248    | 16        |

- **Domain**: Predictive maintenance / Remaining Useful Life (RUL) prediction
- **Exogenous**: Yes — 16 sensor measurements per engine per cycle (multivariate time series)
- **Task**: Regression (predict RUL), not standard point forecast
- **Frequency**: None (cycle-indexed, not calendar-based)
- **Source**: NASA CMAPSS dataset
- **Notes**: This is the only industrial/machinery dataset. Not comparable to energy use case.

---

## 2. Energy-Related Datasets Summary

| Dataset | Energy Type | Freq | n_series | Exogenous | Comparable to EnerCast |
|---------|-------------|------|----------|-----------|------------------------|
| ETTh1/ETTh2 | Electricity transformer (oil temp + load) | Hourly | 1 → 7 vars | Yes (multivariate channels) | Medium — transformer load, not generation |
| ETTm1/ETTm2 | Same at 15min | 15min | 7 | Yes | Medium |
| ECL | Electricity consumption (customers) | 15min→H | 321 | Yes (cross-series) | High — demand forecasting |
| ERCOT | Texas electricity load | Hourly | 1 | No (in datasetsforecast) | High — used in mlforecast tutorial |
| PJM Load | Eastern US electricity load | Hourly | 1 | No | High — used in mlforecast tutorial |

**Key finding**: No wind power, solar generation, or renewable energy datasets exist in
`datasetsforecast`. The closest are electricity consumption and transformer load datasets.

The two most energy-relevant datasets used in mlforecast tutorials are:
- PJM Hourly Load: `https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJM_Load_hourly.csv`
- ERCOT Clean: `https://datasets-nixtla.s3.amazonaws.com/ERCOT-clean.csv`

These are external URLs, not in the datasetsforecast package.

---

## 3. Benchmarks: What Nixtla Provides

### 3.1 Official mlforecast Benchmarks

**There is no dedicated benchmark repository for mlforecast.** The mlforecast repo has:
- `nbs/docs/tutorials/` — tutorials (electricity_load, electricity_peak, m4, prediction_intervals)
- `nbs/docs/how-to-guides/` — how-to guides including `exogenous_features.ipynb`, `mlflow.ipynb`, `cross_validation.ipynb`

Kaggle notebooks with M4 and M5 benchmarks:
- M5: https://www.kaggle.com/code/lemuz90/m5-mlforecast-eval
- M5 (Polars): https://www.kaggle.com/code/lemuz90/m5-mlforecast-eval-polars
- M4: https://www.kaggle.com/code/lemuz90/m4-competition
- M4 with CV: https://www.kaggle.com/code/lemuz90/m4-competition-cv
- VN1 competition: https://colab.research.google.com/drive/1UdhCAk49k6HgMezG-U_1ETnAB5pYvZk9

**No published table comparing mlforecast XGBoost vs manual XGBoost pipelines** exists in the
official repo. Nixtla's pitch for mlforecast is speed and API simplicity, not accuracy gains.

### 3.2 statsforecast Benchmarks (More Comprehensive)

Located at: https://github.com/Nixtla/statsforecast/tree/main/experiments/

| Experiment | What it compares | Dataset | Key result |
|------------|-----------------|---------|------------|
| `arima_prophet_adapter` | AutoARIMA vs FB-Prophet | M3, M4, Tourism, PeytonManning | AutoARIMA: 17% MAPE improvement, 37x faster |
| `m3` | StatsForecast ensemble vs Deep Learning models | M3 (3,003 series) | Statistical ensemble within 0.36 SMAPE of best DL ensemble, 25,000x faster |
| `benchmarks_at_scale` | StatsForecast models at 1M–10M series | Synthetic | 1M series in 5min on 128 CPUs; SeasonalNaive MSE=0.04 |
| `arima_xreg` | AutoARIMA with exogenous regressors | Various | Demonstrates statsforecast exog support |

**Key result from `m3` experiment**:
```
Statistical ensemble (AutoARIMA + ETS + CES + DynamicTheta):
- Runs in 5.6 min on 96 CPUs
- Cost: ~$0.5 USD
- SMAPE competitive with deep learning ensembles (within 0.36 points)

Deep learning ensemble:
- Takes 14+ days to train
- Costs ~$11,000 USD
- Only 0.36 SMAPE better
```

Source: https://github.com/Nixtla/statsforecast/tree/main/experiments/m3

### 3.3 Published Papers / Comparison Methodology

Nixtla references these academic benchmarks:
- **M4 competition** (Makridakis et al.) — standard benchmark, OWA metric
- **Informer paper** (Zhou et al., AAAI 2021) — established ETT/ECL/Traffic/Weather benchmark suite
- **Autoformer** (Wu et al., NeurIPS 2021) — added ILI, extended horizons (96-720)

For comparing ML models specifically, Nixtla's recommended approach is:
1. Use M4 or M5 as benchmark datasets (available via datasetsforecast)
2. Evaluate with sMAPE (M4) or WRMSSE (M5)
3. Use `cross_validation()` for temporal evaluation

**No paper** from Nixtla directly compares mlforecast vs manual XGBoost pipelines.

---

## 4. Exogenous Features — Which Datasets Have Them

| Dataset | Type of Exogenous | Details |
|---------|-------------------|---------|
| **M5** | Rich, real-world exog | sell_prices (weekly), event_name/type (cultural, national, sporting), SNAP benefits flags |
| **LongHorizon (ETT, ECL, Weather, Traffic)** | Multivariate channels as cross-series features | Other variables of same system returned as X_df |
| **PHM2008** | Sensor readings | 16 sensor channels per engine |
| **Favorita (Kaggle full)** | Promotions, oil price, holidays | Not bundled in datasetsforecast |
| **ERCOT/PJM (tutorials)** | None in data | Calendar features generated by mlforecast |
| M3, M4, M3H, Hierarchical | **None** | Univariate only |

**Bottom line for EnerCast**: The only datasetsforecast dataset with real exogenous features
comparable to weather-driven forecasting is M5 (retail, not energy) and the LongHorizon ETT/ECL
datasets (electricity, but multivariate channels not true exogenous weather inputs).

---

## 5. How mlforecast Handles Exogenous Features

From `nbs/docs/how-to-guides/exogenous_features.ipynb`:

```python
from mlforecast import MLForecast

# Static features: replicated automatically from training data
# Dynamic features: must be provided via X_df at predict time
fcst = MLForecast(
    models=lgb.LGBMRegressor(...),
    freq='D',
    lags=[7],
    lag_transforms={1: [ExpandingMean()]},
    date_features=['dayofweek', 'month'],
)

# Fit: pass df with all features (static + dynamic columns alongside y)
fcst.fit(
    series_with_features,
    static_features=['store_id', 'category']  # everything else treated as dynamic
)

# Predict: must supply future dynamic feature values
preds = fcst.predict(h=7, X_df=future_features_df)
```

Three categories:
1. **Static**: declared in `static_features=[]`, replicated for all forecast steps automatically
2. **Dynamic**: any column NOT in `static_features`, requires future values via `X_df`
3. **Calendar**: auto-generated from `date_features=['dayofweek', 'month', 'year', ...]`

Fourier features for multiple seasonalities:
```python
from utilsforecast.feature_engineering import fourier
# generates sin/cos pairs for seasonal decomposition
```

---

## 6. Relevance Assessment for EnerCast

### For wind forecasting validation:
- No direct wind dataset in Nixtla ecosystem
- **Best proxy**: ETTh1/ETTh2 (electricity transformer, hourly, 14400 observations)
  - Has multiple correlated signals (like wind + power relationship)
  - Standard benchmark for hourly energy forecasting
  - Available via `LongHorizon2.load(directory, 'ETTh1')`

### For electricity demand (Spain ENTSO-E):
- **ECL dataset** (321 customers, hourly) is the best Nixtla analog
- **PJM/ERCOT** tutorials are electricity load use cases directly

### For benchmarking methodology:
- Use M4 Hourly group (414 series) as general quality check
- Use ETTh1 for electricity-specific comparison with transformer papers
- Nixtla does NOT provide a head-to-head mlforecast vs hand-coded XGBoost comparison

---

## 7. Key URLs

| Resource | URL |
|----------|-----|
| datasetsforecast GitHub | https://github.com/Nixtla/datasetsforecast |
| datasetsforecast docs | https://nixtlaverse.nixtla.io/datasetsforecast/ |
| mlforecast exogenous features guide | https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/exogenous_features.html |
| mlforecast electricity load tutorial | https://nixtlaverse.nixtla.io/mlforecast/docs/tutorials/electricity_load_forecasting.html |
| mlforecast electricity peak tutorial | https://nixtlaverse.nixtla.io/mlforecast/docs/tutorials/electricity_peak_forecasting.html |
| statsforecast m3 experiment | https://github.com/Nixtla/statsforecast/tree/main/experiments/m3 |
| statsforecast prophet comparison | https://github.com/Nixtla/statsforecast/tree/main/experiments/arima_prophet_adapter |
| Nixtla blog: exogenous variables | https://github.com/Nixtla/blog/blob/main/posts/mlforecast-exogenous-variables.md |
| M4 Kaggle notebook | https://www.kaggle.com/code/lemuz90/m4-competition |
| M5 Kaggle notebook | https://www.kaggle.com/code/lemuz90/m5-mlforecast-eval |
