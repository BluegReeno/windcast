# Forecasting Libraries Landscape — EnerCast Reference

**Last updated**: 2026-04-02  
**Context**: Energy forecasting (wind, solar, demand) — Python ecosystem

---

## Nixtla Stack (statsforecast / mlforecast / neuralforecast)

Three complementary libraries with a shared API convention and Polars support.

### statsforecast

Fast statistical/econometric models. 40+ models.

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive, MSTL

sf = StatsForecast(
    models=[AutoARIMA(), AutoETS(), SeasonalNaive(season_length=24)],
    freq="h",
    n_jobs=-1,
)
sf.fit(df)  # df must have columns: unique_id, ds, y
forecasts = sf.predict(h=24)  # h = forecast horizon
```

Input format: long dataframe with `unique_id`, `ds` (datetime), `y` (target).

**Key models for energy:**
- `AutoARIMA` — auto ARIMA with seasonal detection
- `AutoETS` — auto exponential smoothing (best general-purpose baseline)
- `AutoTheta` / `DynamicOptimizedTheta` — strong for seasonal data
- `MSTL` — multiple seasonality (daily + weekly + annual) — relevant for demand
- `SeasonalNaive(season_length=24)` — seasonal persistence, strong baseline
- `HoltWinters` — classic triple exponential smoothing

Speed: 20x faster than pmdarima, 500x faster than Prophet, 4x faster than statsmodels.  
Distributed: Spark, Dask, Ray. 1M series in 30 min.

### mlforecast

ML models (sklearn-compatible) with automated time-series feature engineering.

```python
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
import lightgbm as lgb

mlf = MLForecast(
    models=[lgb.LGBMRegressor()],
    freq="h",
    lags=[1, 24, 168],
    lag_transforms={1: [("rolling_mean", 24)], 24: [("rolling_std", 24)]},
    date_features=["hour", "dayofweek", "month"],
)
mlf.fit(df)  # same unique_id / ds / y format
mlf.predict(h=24)
```

Key advantage over manual feature engineering: handles recursive multi-step forecasting and lag generation automatically across many series. Works with XGBoost, LightGBM, any sklearn regressor.

### neuralforecast

Neural architectures. Useful for longer horizons and multi-step probabilistic forecasting.

Key models: NBEATS, NHITS, TFT (Temporal Fusion Transformer), PatchTST, LSTM, RNN.  
Same API as statsforecast (`fit` / `predict`), same input format.

---

## Other Libraries

| Library | Install | Best use case | Notes |
|---------|---------|--------------|-------|
| **darts** | `pip install darts` | Unified interface all model types | Supports Chronos-2 foundation models. Heavier dependency. |
| **sktime** | `pip install sktime` | Research, sklearn pipelines | Large API surface, academic use |
| **prophet** | `pip install prophet` | Business seasonality + holidays | 500x slower than statsforecast. Poor fit for wind/solar physics. Avoid. |
| **R fable / forecast** | R only | Gold standard for ARIMA/ETS/Theta | statsforecast is the Python equivalent |

---

## Statistical vs ML — Decision Guide for Energy

### Use statistical models (statsforecast) when:
- Establishing baselines (always do this before ML)
- Short training history (< 1 year)
- No exogenous variables (NWP not available)
- Univariate forecasting: autocorrelation dominates
- Interpretability required
- Probabilistic intervals with statistical guarantees

### Use ML models (XGBoost / LightGBM) when:
- NWP features available (wind speed, direction, temperature, irradiance)
- Non-linear physics interactions matter (v³, turbulence, wake)
- 2+ years of training data
- Cross-turbine or cross-site generalization needed
- Feature engineering can encode domain knowledge

### Typical energy forecasting hierarchy by horizon:

| Horizon | Best approach |
|---------|--------------|
| < 1 hour | Persistence (hard to beat) |
| 1–6 hours | ARIMA/ETS on residuals OR hybrid NWP + statistical correction |
| 6–48 hours | XGBoost / LightGBM with NWP features (production standard) |
| 48h+ | Neural (TFT, NHITS) or ensemble; requires more data |

---

## M-Competition Context (academic validation)

- **M4 winner**: ES-RNN hybrid (exponential smoothing + neural). Key lesson: combining statistical structure with ML flexibility wins.
- **M5 winners**: LightGBM-based solutions dominated for demand forecasting with rich features.
- **General finding**: Pure ML rarely beats well-tuned statistical methods on clean univariate time series. Add NWP covariates and ML dominates.

---

## EnerCast Integration Plan

### Phase 4 demo (minimal, high impact)
Add statsforecast baselines alongside persistence:

```bash
uv add statsforecast
```

```python
# In scripts/train.py or a new scripts/train_baselines.py
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

# Run baselines, log to MLflow, compare with XGBoost results
```

Showing XGBoost + NWP beats AutoARIMA + AutoETS + SeasonalNaive is a strong proof of value.

### Roadmap slide (no implementation needed)
Position mlforecast as the path to scale: "100 sites, same pipeline, automatic feature engineering."

---

## Key Sources
- statsforecast docs: https://nixtlaverse.nixtla.io/statsforecast/
- mlforecast docs: https://nixtlaverse.nixtla.io/mlforecast/
- neuralforecast docs: https://nixtlaverse.nixtla.io/neuralforecast/
- darts docs: https://unit8co.github.io/darts/
- Forecasting: Principles and Practice (Hyndman): https://otexts.com/fpp3/
- IIF OSF Workshop: https://event.nectric.com.au/iif-osf/
