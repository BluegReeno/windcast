# mlforecast Patterns — Implementation Reference

**Version**: 1.0.31 (current on PyPI as of April 2026)
**Docs**: https://nixtlaverse.nixtla.io/mlforecast/
**Source**: https://github.com/Nixtla/mlforecast

---

## 1. MLForecast Constructor

```python
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean, ExponentiallyWeightedMean
from mlforecast.target_transforms import Differences

MLForecast(
    models,                       # single model or list/dict of models (required)
    freq,                         # str | int | pd.BaseOffset (required)
    lags=None,                    # list[int]  — e.g. [1, 6, 24, 48]
    lag_transforms=None,          # dict[int, list[callable]]
    date_features=None,           # list[str | callable]
    num_threads=1,                # int  (-1 = use all CPU cores)
    target_transforms=None,       # list[TargetTransform]
    lag_transforms_namer=None,    # callable — custom feature naming
)
```

### freq values for energy data

```python
# pandas offset aliases (pandas >= 2.2 uses lowercase; both work in practice)
freq='10min'   # 10-minute SCADA  (old alias: '10T')
freq='15min'   # 15-minute data   (old alias: '15T')
freq='1h'      # hourly           (old alias: 'H')
freq='D'       # daily
freq='MS'      # month start
freq=1         # integer-indexed series (no real timestamps, use when ds is int)
```

### Models: XGBoost and LightGBM

```python
import xgboost as xgb
import lightgbm as lgb

# Single model
fcst = MLForecast(models=xgb.XGBRegressor(n_estimators=500, learning_rate=0.05), freq='1h', ...)

# Multiple models (list — auto-named by class name)
fcst = MLForecast(
    models=[
        xgb.XGBRegressor(n_estimators=500, learning_rate=0.05),
        lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbosity=-1),
    ],
    freq='1h',
)

# Multiple models (dict — explicit names)
fcst = MLForecast(
    models={
        'xgb': xgb.XGBRegressor(n_estimators=500),
        'lgb_q50': lgb.LGBMRegressor(objective='quantile', alpha=0.5, verbosity=-1),
        'lgb_q90': lgb.LGBMRegressor(objective='quantile', alpha=0.9, verbosity=-1),
    },
    freq='10min',
)
```

### Lags and lag_transforms

```python
fcst = MLForecast(
    models=xgb.XGBRegressor(),
    freq='1h',
    lags=[1, 2, 3, 6, 12, 24, 48, 168],   # lag 1h to lag 1 week
    lag_transforms={
        1:  [ExpandingMean()],
        6:  [RollingMean(window_size=24)],
        24: [RollingMean(window_size=48), ExponentiallyWeightedMean(alpha=0.3)],
    },
    target_transforms=[Differences([24])],   # seasonal differencing
)
```

---

## 1b. lag_transforms API Reference (verified from source)

**Import path**: `from mlforecast.lag_transforms import <ClassName>`

All classes listed in `__all__`:

```
RollingMean, RollingStd, RollingMin, RollingMax, RollingQuantile
SeasonalRollingMean, SeasonalRollingStd, SeasonalRollingMin, SeasonalRollingMax, SeasonalRollingQuantile
ExpandingMean, ExpandingStd, ExpandingMin, ExpandingMax, ExpandingQuantile
ExponentiallyWeightedMean
Offset, Combine
```

### RollingMean / RollingStd / RollingMin / RollingMax

All four share the same constructor via `_RollingBase`:

```python
RollingMean(
    window_size: int,                         # number of samples in window (required)
    min_samples: Optional[int] = None,        # min samples to compute; defaults to window_size
    global_: bool = False,                    # aggregate across all series by timestamp
    groupby: Optional[Sequence[str]] = None,  # group by static feature columns
)
```

### RollingQuantile

```python
RollingQuantile(
    p: float,              # quantile (0.0 to 1.0) e.g. p=0.9 for P90
    window_size: int,
    min_samples: Optional[int] = None,
    global_: bool = False,
    groupby: Optional[Sequence[str]] = None,
)
```

### SeasonalRollingMean / SeasonalRollingStd / etc.

```python
SeasonalRollingMean(
    season_length: int,    # periodicity (e.g. 144 for 24h at 10-min resolution)
    window_size: int,      # number of seasonal periods in window
    min_samples: Optional[int] = None,
    global_: bool = False,
    groupby: Optional[Sequence[str]] = None,
)
```

### ExpandingMean / ExpandingStd / ExpandingMin / ExpandingMax

No required arguments:

```python
ExpandingMean(
    global_: bool = False,
    groupby: Optional[Sequence[str]] = None,
)
```

### ExpandingQuantile

```python
ExpandingQuantile(
    p: float,              # required
    global_: bool = False,
    groupby: Optional[Sequence[str]] = None,
)
```

### ExponentiallyWeightedMean

```python
ExponentiallyWeightedMean(
    alpha: float,          # smoothing factor, 0 < alpha < 1 (required)
    global_: bool = False,
    groupby: Optional[Sequence[str]] = None,
)
```

### Offset — shift series before computing transformation

```python
Offset(
    tfm: _BaseLagTransform,   # transformation to apply
    n: int,                    # positions to shift before applying
)
# Example: Offset(RollingMean(window_size=10), 2) at lag 5 → computes rolling mean at lag 7
```

### Combine — binary operator on two transforms

```python
import operator
Combine(
    tfm1: _BaseLagTransform,
    tfm2: _BaseLagTransform,
    operator: Callable,         # e.g. operator.truediv, operator.add
)
# Example: Combine(Lag(1), Lag(2), operator.truediv)  → lag1 / lag2
```

### Feature naming convention

Generated feature names follow the pattern: `{transform_name}_lag{N}_{changed_params}`

```python
RollingMean(window_size=24)       # at lag 6:  "rolling_mean_lag6_window_size24"
RollingMean(window_size=10)       # default min_samples → "rolling_mean_lag5_window_size10"
ExponentiallyWeightedMean(0.3)    # at lag 1:  "exponentially_weighted_mean_lag1_alpha0.3"
ExpandingMean()                   # at lag 1:  "expanding_mean_lag1"
```

---

## 2. Expected DataFrame Format

Long format with at minimum these three columns:

```
unique_id | ds                  | y       | [exogenous_col_1] | [exogenous_col_2] | ...
----------|---------------------|---------|-------------------|-------------------|----
turbine_1 | 2023-01-01 00:00:00 | 1520.4  | 8.2               | 12.1              |
turbine_1 | 2023-01-01 00:10:00 | 1618.3  | 8.5               | 12.0              |
turbine_2 | 2023-01-01 00:00:00 | 984.1   | 6.1               | 11.8              |
```

- `unique_id`: series identifier (string). For a single turbine use a constant e.g. `"T001"`.
- `ds`: timestamp (datetime64 for real timestamps, int for integer-indexed).
- `y`: target (power output in kW/MW).
- Additional columns: treated as exogenous features automatically.

---

## 3. fit() Method

```python
MLForecast.fit(
    df,                          # pd.DataFrame or pl.DataFrame (required)
    id_col='unique_id',          # str
    time_col='ds',               # str
    target_col='y',              # str
    static_features=None,        # list[str] — columns that do NOT vary over time
    max_horizon=None,            # int — enables DIRECT forecasting (one model per step)
    horizons=None,               # list[int] — train only for specific steps (sparse direct)
    fitted=False,                # bool — save in-sample predictions
    as_numpy=False,              # bool
    weight_col=None,             # str — sample weights column
    **kwargs                     # passed to underlying model fit()
) -> MLForecast
```

### Exogenous features in fit()

Any column in `df` beyond `unique_id`, `ds`, `y` is automatically treated as a feature.
Use `static_features` to flag columns that are constant per series (they will be replicated
for each forecast step instead of requiring future values).

```python
# df has columns: unique_id, ds, y, wind_speed_m_s, wind_dir_deg, temp_c, turbine_model
fcst.fit(
    df,
    static_features=['turbine_model'],   # constant per turbine — auto-replicated
    # wind_speed_m_s, wind_dir_deg, temp_c → dynamic exogenous, need X_df at predict time
)
```

Inspect which features were used:
```python
fcst.ts.features_order_   # list of all feature column names fed to the model
```

---

## 4. predict() Method

```python
MLForecast.predict(
    h,                              # int — horizon length (required)
    before_predict_callback=None,   # callable
    after_predict_callback=None,    # callable
    new_df=None,                    # pd.DataFrame | pl.DataFrame — new observations for transfer
    level=None,                     # list[int | float] — confidence levels e.g. [80, 90]
    X_df=None,                      # pd.DataFrame | pl.DataFrame — future exogenous values
    ids=None,                       # list[str] — subset of series to forecast
) -> pd.DataFrame | pl.DataFrame
```

### X_df format

When dynamic exogenous features were used in training, `X_df` must be provided:

```python
# X_df: future values of dynamic features, one row per (series, future_timestep)
# Must contain: unique_id, ds, and all dynamic exogenous columns
# Must cover exactly h timesteps per series

future_exog = pd.DataFrame({
    'unique_id': ['T001'] * 24 + ['T002'] * 24,
    'ds': pd.date_range('2024-01-02', periods=24, freq='1h').tolist() * 2,
    'wind_speed_m_s': nwp_wind_speed_forecast,   # from NWP model
    'temp_c': nwp_temp_forecast,
})

preds = fcst.predict(h=24, X_df=future_exog)
```

Output columns: `unique_id`, `ds`, one column per model name.

---

## 5. Multi-Step Forecasting: Recursive vs Direct

### Recursive (default)

The model is called once and re-uses its own predictions as lagged inputs for subsequent steps.
Fast — only one model trained per estimator type.

```python
fcst.fit(df)                     # no max_horizon
preds = fcst.predict(h=24)       # returns 24 rows per series
```

### Direct (one model per horizon step)

Trains N separate models. Each model is specialized for its specific horizon.
Eliminates error accumulation. Slower and more memory-intensive.

```python
# Train 24 models (one per step from h=1 to h=24)
fcst.fit(df, max_horizon=24)
preds = fcst.predict(h=24)       # each row uses its dedicated model
```

### Sparse direct (specific horizons only)

When you only need forecasts at specific intervals (e.g., 1h, 6h, 24h ahead):

```python
# For a 10-min dataset: h=6 → 1h ahead, h=36 → 6h ahead, h=144 → 24h ahead
fcst.fit(df, horizons=[6, 36, 144])
preds = fcst.predict(h=144)      # returns only 3 rows per series (steps 6, 36, 144)
```

Note: `max_horizon` and `horizons` are mutually exclusive.

---

## 6. Exogenous / Pre-Computed Features — Full Pattern

This is the key pattern for WindCast: add NWP features (wind_speed, temp) as columns in df.

```python
import pandas as pd
import xgboost as xgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import Differences

# --- Training data ---
# df already has pre-computed features: wind_speed_cubed, wind_dir_sin, wind_dir_cos, temp_c
# turbine_rated_power is static (constant per turbine)
train_df = pd.DataFrame({
    'unique_id': ...,
    'ds': ...,
    'y': ...,                        # power_kw
    'wind_speed_m_s': ...,           # NWP or SCADA wind
    'wind_speed_cubed': ...,         # pre-computed: v^3
    'wind_dir_sin': ...,             # pre-computed: sin(dir_rad)
    'wind_dir_cos': ...,             # pre-computed: cos(dir_rad)
    'temp_c': ...,                   # NWP temperature
    'turbine_rated_power_kw': ...,   # static — same value for all rows of same turbine
})

fcst = MLForecast(
    models=xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, tree_method='hist'),
    freq='10min',
    lags=[1, 6, 12, 144],           # lag 10min, 1h, 2h, 24h
    lag_transforms={
        6:   [RollingMean(window_size=36)],    # 6h rolling mean
        144: [RollingMean(window_size=144)],   # 24h rolling mean
    },
    target_transforms=[Differences([144])],    # 24h seasonal differencing
    num_threads=-1,
)

fcst.fit(
    train_df,
    static_features=['turbine_rated_power_kw'],   # replicated at predict time automatically
)

# --- Future data for prediction ---
# X_df contains NWP forecasts for the horizon window
future_df = pd.DataFrame({
    'unique_id': ...,
    'ds': ...,
    'wind_speed_m_s': nwp_wind_speed,
    'wind_speed_cubed': nwp_wind_speed ** 3,
    'wind_dir_sin': np.sin(np.radians(nwp_wind_dir)),
    'wind_dir_cos': np.cos(np.radians(nwp_wind_dir)),
    'temp_c': nwp_temp,
    # turbine_rated_power_kw NOT needed — it's static and auto-replicated
})

preds = fcst.predict(h=144, X_df=future_df)   # 24h ahead at 10-min resolution
```

---

## 7. Cross-Validation

Built-in temporal CV with `cross_validation()`. Uses expanding window by default.

```python
cv_results = fcst.cross_validation(
    df=train_df,
    h=144,           # horizon per window (e.g. 24h at 10-min = 144 steps)
    n_windows=3,     # number of CV windows
    # step_size=h    # gap between window cutoffs (default = h, no overlap)
)
# Returns: unique_id, ds, cutoff, y, {model_name}

# Evaluate
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, smape

scores = evaluate(
    cv_results.drop(columns='cutoff'),
    metrics=[mae, rmse, smape],
    agg_fn='mean',
)
```

Cross-validation works with `max_horizon` and `horizons` as well:

```python
cv_results = fcst.cross_validation(
    df=train_df,
    h=144,
    n_windows=3,
    horizons=[6, 36, 144],   # only evaluate at 1h, 6h, 24h ahead
)
```

---

## 8. Polars Support

Polars DataFrames work natively — no conversion needed. Pass a `pl.DataFrame` to `fit()` and
`predict()` returns a `pl.DataFrame`.

**Polars is an optional dependency.** Install with:
```bash
uv add "mlforecast[polars]"   # installs polars[numpy]
# or: polars>=1.0 already in pyproject.toml deps — mlforecast auto-detects it
```

Polars support was added in **v0.11.0** (Nov 2023). Current stable: **v1.0.31** (Mar 2026).

```python
import polars as pl
from mlforecast import MLForecast
import xgboost as xgb

df_pl = pl.DataFrame({
    'unique_id': ['T001'] * n,
    'ds': pl.Series(timestamps),    # must be pl.Datetime dtype
    'y': power_values,
    'wind_speed_m_s': wind_values,
})

fcst = MLForecast(
    models=xgb.XGBRegressor(),
    freq='10min',          # Polars uses duration strings: '10m', '1h', '1d' — NOT pandas aliases
    lags=[1, 6, 144],
)

fcst.fit(df_pl, as_numpy=True)       # REQUIRED for LightGBM with Polars (see gotchas)
preds = fcst.predict(h=144)          # returns pl.DataFrame when input was pl.DataFrame
```

Important: the `ds` column must be `pl.Datetime` (not string). Cast if needed:
```python
df_pl = df_pl.with_columns(pl.col('ds').cast(pl.Datetime('us')))
```

**freq format differs between Polars and pandas DataFrames:**
```python
# For pd.DataFrame:
freq='10min'   # or '10T' (deprecated)
freq='1h'      # or 'H' (deprecated)
freq='D'

# For pl.DataFrame — use Polars duration strings:
freq='10m'     # 10 minutes
freq='1h'      # 1 hour
freq='1d'      # 1 day
```

The underlying ML models (XGBoost, LightGBM) receive numpy arrays — the Polars/pandas
conversion is handled internally by mlforecast via `utilsforecast`.

---

## 9. Complete Minimal Example (XGBoost + Exogenous + Multi-Horizon)

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import Differences

# --- Synthetic wind power data ---
np.random.seed(42)
n = 8640  # 60 days at 10-min resolution
timestamps = pd.date_range('2023-01-01', periods=n, freq='10min')
wind_speed = np.clip(np.random.normal(8, 3, n), 0, 25)
power = np.clip((wind_speed / 12) ** 3 * 2000, 0, 2000) + np.random.normal(0, 50, n)

train_df = pd.DataFrame({
    'unique_id': 'T001',
    'ds': timestamps,
    'y': power,
    'wind_speed_m_s': wind_speed,
    'wind_speed_cubed': wind_speed ** 3,
    'wind_dir_sin': np.sin(np.random.uniform(0, 2 * np.pi, n)),
    'wind_dir_cos': np.cos(np.random.uniform(0, 2 * np.pi, n)),
})

# --- Model ---
fcst = MLForecast(
    models=xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        tree_method='hist',
        random_state=42,
    ),
    freq='10min',
    lags=[1, 6, 12, 36, 144],             # 10min, 1h, 2h, 6h, 24h
    lag_transforms={
        6:   [RollingMean(window_size=18)],  # 3h rolling mean at lag 1h
        144: [RollingMean(window_size=144)], # 24h rolling mean at lag 24h
    },
    target_transforms=[Differences([144])],  # 24h differencing
    num_threads=-1,
)

# --- Fit (recursive strategy) ---
fcst.fit(train_df)

# --- Future NWP values for 24h horizon ---
future_wind = np.clip(np.random.normal(9, 3, 144), 0, 25)
future_df = pd.DataFrame({
    'unique_id': 'T001',
    'ds': pd.date_range(timestamps[-1] + pd.Timedelta('10min'), periods=144, freq='10min'),
    'wind_speed_m_s': future_wind,
    'wind_speed_cubed': future_wind ** 3,
    'wind_dir_sin': np.sin(np.random.uniform(0, 2 * np.pi, 144)),
    'wind_dir_cos': np.cos(np.random.uniform(0, 2 * np.pi, 144)),
})

# --- Predict 24h ahead (144 steps at 10-min) ---
preds = fcst.predict(h=144, X_df=future_df)
print(preds.head())
# unique_id | ds                  | XGBRegressor
# T001      | 2023-03-02 00:10:00 | 1432.1
# T001      | 2023-03-02 00:20:00 | 1489.7
# ...

# --- Cross-validation ---
cv = fcst.cross_validation(df=train_df, h=144, n_windows=3)
print(cv.head())
```

---

## 10. Key Gotchas for WindCast

| Issue | Solution |
|-------|----------|
| `X_df` must cover exactly `h` steps per series | Generate future timestamps from `last_train_ts + timedelta` |
| Static features (e.g. rated power) NOT needed in `X_df` | They are auto-replicated from training data |
| `ds` column must be sorted within each `unique_id` | Sort before fitting |
| `Differences([144])` requires 144+ rows to recover first values | Keep enough history or use `keep_last_n` in fit |
| Polars `ds` must be `pl.Datetime`, not string | `pl.col('ds').cast(pl.Datetime('us'))` |
| **LightGBM + Polars DataFrame requires `as_numpy=True`** | `fcst.fit(df_pl, as_numpy=True)` — confirmed in test suite comment: "LightGBM with polars DataFrames not fully supported yet" |
| Polars freq strings differ from pandas aliases | `'10m'` not `'10min'`; `'1h'` same; `'1d'` not `'D'` — use Polars duration format |
| pandas >= 2.2 deprecates `'T'` and `'H'` aliases | Use `'min'`/`'10min'` and `'h'`/`'1h'` instead |
| `max_horizon=N` trains N models — can be slow for large N | Use `horizons=[6, 36, 144]` for sparse direct |
| Cross-validation with exogenous requires X_df-compatible columns in df | Pass full df (not split) to `cross_validation()` |
| Integer `freq=1` is for integer-indexed `ds` only | Use pandas offset string when `ds` is datetime |
