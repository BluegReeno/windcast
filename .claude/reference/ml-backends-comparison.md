# ML Backends Comparison — XGBoost Manual vs mlforecast

## Two Training Pipelines

EnerCast now has two ML training backends. Both log to MLflow for direct comparison.

### 1. Traditional XGBoost (`scripts/train.py`)

**How it works:**
- Feature engineering is manual: we compute lags, rolling stats, domain features in Polars
- One XGBoost model trained per horizon via `shift(-h)` target
- Each horizon is independent — no recursive prediction

**Strengths:**
- Full control over every feature
- Easy to debug (features are explicit columns)
- Well-tested (201 tests before mlforecast integration)

**Weaknesses:**
- No recursive prediction: at horizon h>1, uses actual lag-1 value instead of model's own h-1 prediction. This is incorrect for production multi-step forecasting.
- Manual multi-horizon loop: N horizons = N separate train calls
- ~200 lines of feature plumbing code

**Data flow:**
```
data/processed/*.parquet → build_features.py → data/features/*.parquet → train.py → MLflow
```

### 2. mlforecast / Nixtla (`scripts/train_mlforecast.py`)

**How it works:**
- We compute only domain-specific exogenous features (V³, HDD/CDD, clearsky_ratio, cyclic encoding)
- mlforecast handles all lag/rolling feature generation internally
- Supports recursive, direct, and sparse direct multi-step strategies

**Strengths:**
- Correct recursive prediction (model's own predictions feed into next step's lags)
- Sparse direct strategy: trains exactly N models for N horizons (e.g., 5 for [1,6,12,24,48])
- Less code to maintain (~50 lines vs ~200 for manual lags/rolling)
- Built-in temporal cross-validation

**Weaknesses:**
- Less transparency (lag features are generated internally)
- Exogenous features need future values at predict time (`X_df`) — requires NWP forecasts
- Less mature in our codebase (fewer tests, less battle-tested)

**Data flow:**
```
data/processed/*.parquet → train_mlforecast.py (builds exogenous + delegates lags) → MLflow
```

## Feature Sets

| Pipeline | Feature Set Pattern | Example |
|----------|-------------------|---------|
| XGBoost manual | `wind_baseline` / `wind_enriched` / `wind_full` | Includes lags + rolling + domain |
| mlforecast | `wind_exog_baseline` / `wind_exog_enriched` / `wind_exog_full` | Domain features only (no lags/rolling) |

## Comparison Strategy

To compare the two backends fairly:

1. **Same data split**: Both use `_temporal_split()` with same train/val years
2. **Same MLflow experiment**: Use different run names but same experiment for side-by-side
3. **Same metrics**: Both compute MAE, RMSE, bias, skill score via `compute_metrics()`
4. **Same horizons**: Both use `forecast_horizons = [1, 6, 12, 24, 48]`

### Running a Comparison

```bash
# Traditional XGBoost
uv run python scripts/train.py --domain wind --turbine-id kwf1 --feature-set wind_enriched

# mlforecast (sparse direct)
uv run python scripts/train_mlforecast.py --domain wind --turbine-id kwf1 --feature-set wind_exog_enriched

# Compare in MLflow
mlflow ui
```

### What to Compare

| Metric | Why |
|--------|-----|
| MAE per horizon | Core accuracy metric |
| RMSE per horizon | Penalizes large errors |
| Skill score vs persistence | Both should beat persistence; which beats it more? |
| Training time | mlforecast may be faster (fewer models with sparse direct) |
| Horizon degradation | How fast does accuracy degrade with horizon? Recursive vs direct matters here |

### Expected Differences

- **Short horizons (h=1)**: Should be similar — both have same lag-1 information
- **Long horizons (h=24, h=48)**: mlforecast recursive should be better (uses own predictions as lags) vs XGBoost manual (uses actual lag-1, which is stale at h=48)
- **Sparse direct**: May sacrifice some accuracy at specific horizons for efficiency

## Nixtla Ecosystem — Datasets & Benchmarks

### Available Datasets (`datasetsforecast`)

| Dataset | Domain | Frequency | Series | Exogenous? | Relevance |
|---------|--------|-----------|--------|------------|-----------|
| M3 | Mixed | Y/Q/M/D | 3,003 | No | Low — no energy |
| M4 | Mixed | Y/Q/M/W/D/H | 100,000 | No | Low — scale test only |
| **M5** | **Retail (Walmart)** | **Daily** | **30,490** | **Yes (prices, calendar, events)** | **Medium — best exog test** |
| **LongHorizon (ETT/ECL)** | **Electricity/Weather** | **H/15min** | **1–862** | **Yes (cross-series)** | **High — closest to energy** |
| Hierarchical | Tourism/Traffic | M/Q/D | Variable | No | Low |
| PHM2008 | Predictive maintenance | Cycles | 100–260 | Yes (16 sensors) | Low |

### No Official Benchmarks for mlforecast vs Manual XGBoost

Nixtla does **not** provide accuracy benchmarks for mlforecast. Their benchmarks focus on:
- `statsforecast` vs Deep Learning (statsforecast ensemble is 25,000x faster, comparable accuracy)
- `statsforecast` vs Prophet (AutoARIMA is +17% MAPE, 37x faster)
- Scaling to 1M series in 5 minutes

**The accuracy comparison is ours to build.** This is actually valuable — our energy domain comparison could be contributed back to the community.

### Energy-Adjacent Datasets in LongHorizon

The most relevant for our comparison:
- **ETTh1/ETTh2**: Electricity transformer temperature, hourly, 7 variables. Standard benchmark in time series papers.
- **ECL**: 321 electricity clients, 15min→hourly. Large multivariate electricity dataset.

Neither has wind/solar, but ECL is load forecasting — directly comparable to our demand domain.

### Datasets with Rich Exogenous Features

Only **M5** has truly rich exogenous features (prices, SNAP benefits, calendar events). This mirrors our pattern of domain features + external regressors. Consider using M5 to validate our exogenous pipeline against Nixtla's published results.

## Next Steps for Comparison

1. **Phase 1**: Run both pipelines on wind domain (Kelmarsh), compare metrics in MLflow
2. **Phase 2**: Run both on demand domain (Spain ENTSO-E), compare
3. **Phase 3**: Optionally run on ECL/ETT datasets to benchmark against published results
4. **Phase 4**: Document findings for WN presentation
