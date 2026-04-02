# Feature: Integrate mlforecast (Nixtla) as ML Training Backend

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Add [mlforecast](https://github.com/Nixtla/mlforecast) (Nixtla) as an alternative ML training backend alongside our manual XGBoost pipeline. mlforecast handles lag/rolling feature generation, recursive/direct multi-step prediction, and temporal cross-validation — replacing ~200 lines of manual code while correctly handling recursive feature updates during prediction.

The existing XGBoost pipeline remains untouched. mlforecast is additive.

## User Story

As an energy ML engineer
I want to train models via mlforecast with automatic lag management and multi-horizon strategies
So that I get correct recursive predictions and can scale to many series without manual feature plumbing

## Problem Statement

Our current pipeline manually computes lags/rolling features (in `build_features.py`) then trains one XGBoost per horizon with `shift(-h)` targets. This has two problems:
1. **No recursive prediction**: at horizon h>1, we use the actual lag-1 value instead of the model's own h-1 prediction. This is incorrect for production use.
2. **Manual multi-horizon loop**: training N horizons requires N separate train calls with manual target shifting.

mlforecast solves both: it handles recursive feature updates and supports direct/recursive/sparse-direct strategies natively.

## Solution Statement

Add a `models/mlforecast_model.py` wrapper and a `scripts/train_mlforecast.py` script that:
1. Reads post-QC processed data (not pre-computed features)
2. Computes only domain-specific exogenous features (V³, clearsky_ratio, HDD/CDD, cyclic encoding)
3. Delegates lag/rolling generation to mlforecast
4. Trains with sparse direct strategy matching our horizon config
5. Logs to MLflow
6. Outputs predictions compatible with our existing evaluation pipeline

## Feature Metadata

**Feature Type**: New Capability (additive — existing pipeline untouched)
**Estimated Complexity**: Medium
**Primary Systems Affected**: `models/`, `features/`, `scripts/`
**Dependencies**: `mlforecast>=1.0`, `utilsforecast` (transitive)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — MUST READ BEFORE IMPLEMENTING

- `src/windcast/models/xgboost_model.py` — Existing XGBoost wrapper pattern to mirror (Pydantic config, MLflow logging)
- `src/windcast/models/persistence.py` — Baseline model pattern
- `src/windcast/models/__init__.py` — Package exports to update
- `src/windcast/models/evaluation.py` — `compute_metrics()` and `compute_skill_score()` — reuse for mlforecast evaluation
- `scripts/train.py` — Current training script: temporal split, DOMAIN_CONFIG dict, MLflow run structure, `_build_horizon_target()`. The new script mirrors this structure.
- `scripts/evaluate.py` — Current eval script: loads models from MLflow child runs. mlforecast eval will be different (predict from fitted object, not load per-horizon models).
- `scripts/build_features.py` — Current feature builder. mlforecast needs a lighter version (domain features only, no lags/rolling).
- `src/windcast/features/registry.py` — FeatureSet dataclass, FEATURE_REGISTRY dict. Add new "exogenous-only" feature sets.
- `src/windcast/features/wind.py` — `build_wind_features()`: lags at L62-69, rolling at L73-98, domain-specific at L109-121. mlforecast needs only the domain-specific part.
- `src/windcast/features/demand.py` — Same pattern: lags at L63-71, domain features at L103-145.
- `src/windcast/features/solar.py` — Same pattern: lags at L62-69, domain features at L102-145.
- `src/windcast/config.py` — `WindCastSettings`, `DOMAIN_CONFIG` pattern in train.py, `forecast_horizons` default `[1, 6, 12, 24, 48]`
- `src/windcast/tracking/mlflow_utils.py` — `setup_mlflow()`, `log_feature_set()`, `log_evaluation_results()`, `log_dataframe_artifact()`
- `tests/models/test_xgboost_model.py` — Test pattern: `_make_regression_data()` helper, mock MLflow, assert predictions
- `tests/models/test_evaluation.py` — Metric assertion patterns
- `tests/models/test_persistence.py` — Simplest model test pattern

### New Files to Create

- `src/windcast/models/mlforecast_model.py` — MLForecast wrapper with fit/predict/cross_validate
- `src/windcast/features/exogenous.py` — Domain-specific exogenous feature builders (no lags/rolling)
- `scripts/train_mlforecast.py` — Training CLI using mlforecast
- `tests/models/test_mlforecast_model.py` — Unit tests for mlforecast wrapper

### Relevant Documentation

**mlforecast (Nixtla):**
- [MLForecast class reference](https://nixtlaverse.nixtla.io/mlforecast/forecast.html)
- [Exogenous features guide](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/exogenous_features.html)
- [One model per horizon (direct)](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/one_model_per_horizon.html)
- [Cross-validation guide](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/cross_validation.html)
- Local reference: `.claude/reference/mlforecast-patterns.md`

### Patterns to Follow

**Naming Conventions:**
- Files: `snake_case.py`
- Classes: `PascalCase` (e.g., `MLForecastConfig`)
- Functions: `snake_case` (e.g., `train_mlforecast`)
- Constants: `UPPER_SNAKE_CASE`

**Config Pattern (from xgboost_model.py):**
```python
class XGBoostConfig(BaseModel):
    objective: str = "reg:squarederror"
    n_estimators: int = 500
    ...
```

**MLflow Logging Pattern (from train.py):**
```python
with mlflow.start_run(run_name=f"{run_label}-{feature_set}"):
    mlflow.log_params({...})
    log_feature_set(feature_set, available_cols)
    # nested runs per horizon
    with mlflow.start_run(run_name=f"h{h:02d}", nested=True):
        ...
```

**DOMAIN_CONFIG Pattern (from train.py:24-27):**
```python
DOMAIN_CONFIG: dict[str, dict[str, str]] = {
    "wind": {"target": "active_power_kw", "group": "turbine_id", "lag1": "active_power_kw_lag1"},
    "demand": {"target": "load_mw", "group": "zone_id", "lag1": "load_mw_lag1"},
    "solar": {"target": "power_kw", "group": "system_id", "lag1": "power_kw_lag1"},
}
```

**Data Format — Current (Polars DataFrame, wide):**
```
timestamp_utc | turbine_id | active_power_kw | wind_speed_ms | ... | active_power_kw_lag1 | ...
```

**Data Format — mlforecast (Polars DataFrame, long with required columns):**
```
unique_id | ds                  | y       | wind_speed_ms | wind_dir_sin | wind_speed_cubed | ...
kwf1      | 2016-01-01 00:00:00 | 1520.4  | 8.2           | 0.67         | 551.4            | ...
```
- `unique_id` = group col (turbine_id / zone_id / system_id)
- `ds` = timestamp_utc
- `y` = target col (active_power_kw / load_mw / power_kw)
- remaining cols = exogenous features (domain-specific, NOT lags/rolling)

**Horizon Mapping — steps to time depends on data resolution:**
| Domain | Resolution | Step 1 = | Our horizons [1,6,12,24,48] mean |
|--------|-----------|----------|----------------------------------|
| Wind | 10 min | 10 min | 10m, 1h, 2h, 4h, 8h |
| Demand | 1 hour | 1 hour | 1h, 6h, 12h, 24h, 48h |
| Solar | 15 min | 15 min | 15m, 1.5h, 3h, 6h, 12h |

mlforecast's `horizons` parameter uses step counts, so our existing `forecast_horizons` config maps directly.

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — Dependency + Config

Add mlforecast dependency and create the configuration model.

### Phase 2: Exogenous Feature Extraction

Create functions that compute domain-specific features WITHOUT lags/rolling (those are handled by mlforecast). Also create feature set definitions for the exogenous-only columns.

### Phase 3: MLForecast Model Wrapper

Create the core wrapper: data preparation, fit, predict, MLflow logging.

### Phase 4: Training Script

Create `train_mlforecast.py` CLI script mirroring the structure of `train.py`.

### Phase 5: Testing

Unit tests for the wrapper + exogenous features.

### Phase 6: Validation

Run full validation suite + manual test on wind domain.

---

## STEP-BY-STEP TASKS

### Task 1: ADD mlforecast dependency

- **IMPLEMENT**: Add `"mlforecast[polars]>=1.0"` to `pyproject.toml` dependencies, then run `uv sync`. The `[polars]` extra ensures `polars[numpy]` is pulled in (already satisfied by our existing polars dep but explicit is better).
- **PATTERN**: Follow existing dependency format in `pyproject.toml:13-30`
- **VALIDATE**: `uv run python -c "from mlforecast import MLForecast; from mlforecast.lag_transforms import RollingMean; print('OK')"`

### Task 2: CREATE `src/windcast/features/exogenous.py`

- **IMPLEMENT**: Domain-specific feature builders that produce exogenous columns only (no lags, no rolling). Three functions:
  - `build_wind_exogenous(df: pl.DataFrame, feature_set: str) -> pl.DataFrame` — QC filter, sort, add: wind_dir_sin/cos, wind_speed_cubed, turbulence_intensity, wind_dir_sector, cyclic hour/calendar. Based on which set (baseline/enriched/full).
  - `build_demand_exogenous(df: pl.DataFrame, feature_set: str) -> pl.DataFrame` — QC filter, sort, add: cyclic calendar, HDD/CDD, temperature_c, wind_speed_ms, humidity_pct, price lags (shift over group), is_holiday.
  - `build_solar_exogenous(df: pl.DataFrame, feature_set: str) -> pl.DataFrame` — QC filter, sort, add: cyclic hour, clearsky_ratio, GHI, wind, cyclic calendar.
- **PATTERN**: Mirror `features/wind.py` structure but skip `_add_lag_features()` and `_add_rolling_features()` calls
- **IMPORTS**: `polars`, `math`, `windcast.data.schema.QC_OK`
- **GOTCHA**: Price lag features in demand (`price_lag1`, `price_lag24`) are NOT target lags — they're exogenous lags that mlforecast won't auto-generate. Keep these in the exogenous builder.
- **GOTCHA**: `is_holiday` must be cast to `pl.Int8` for XGBoost compatibility (same as `demand.py:57`)
- **VALIDATE**: `uv run ruff check src/windcast/features/exogenous.py && uv run pyright src/windcast/features/exogenous.py`

### Task 3: ADD exogenous feature set definitions to registry

- **IMPLEMENT**: Add new FeatureSet entries to `features/registry.py` for exogenous-only columns. These list the columns that the exogenous builders produce (NOT lag/rolling cols). Example:
  ```
  WIND_EXOG_BASELINE = FeatureSet(
      name="wind_exog_baseline",
      columns=["wind_speed_ms", "wind_dir_sin", "wind_dir_cos"],
      description="Wind exogenous features for mlforecast (baseline)",
  )
  ```
  Create 3 per domain (baseline/enriched/full) = 9 new entries.
- **PATTERN**: Mirror existing `WIND_BASELINE`, `DEMAND_BASELINE`, etc. in `registry.py:15-160`
- **GOTCHA**: These lists must exactly match the columns produced by the exogenous builders. Cross-validate.
- **VALIDATE**: `uv run python -c "from windcast.features.registry import get_feature_set; print(get_feature_set('wind_exog_baseline'))"`

### Task 4: UPDATE `src/windcast/features/__init__.py`

- **IMPLEMENT**: Export the three new exogenous builder functions
- **PATTERN**: Follow existing exports at `__init__.py:1-21`
- **VALIDATE**: `uv run python -c "from windcast.features import build_wind_exogenous"`

### Task 5: CREATE `src/windcast/models/mlforecast_model.py`

This is the core file. It wraps MLForecast for our multi-domain energy forecasting use case.

- **IMPLEMENT**:

  1. `MLForecastConfig(BaseModel)` — Pydantic config:
     ```python
     class MLForecastConfig(BaseModel):
         """Configuration for mlforecast training."""
         # XGBoost hyperparameters (passed to XGBRegressor)
         n_estimators: int = 500
         learning_rate: float = 0.05
         max_depth: int = 6
         min_child_weight: int = 10
         subsample: float = 0.8
         colsample_bytree: float = 0.8
         # mlforecast settings
         strategy: str = "sparse_direct"  # "recursive" | "direct" | "sparse_direct"
         n_cv_windows: int = 3
     ```

  2. `DomainMLForecastConfig` — domain-specific lag/rolling configuration:
     ```python
     DOMAIN_MLFORECAST: dict[str, dict] = {
         "wind": {
             "target": "active_power_kw",
             "group": "turbine_id",
             "freq": "10m",  # Polars duration format, NOT '10min'
             "lags": [1, 2, 3, 6, 12, 24],
             "rolling_windows": [6, 12, 24],
         },
         "demand": {
             "target": "load_mw",
             "group": "zone_id",
             "freq": "1h",
             "lags": [1, 2, 24, 168],
             "rolling_windows": [24, 168],
         },
         "solar": {
             "target": "power_kw",
             "group": "system_id",
             "freq": "15m",  # Polars duration format, NOT '15min'
             "lags": [1, 2, 4, 8, 96],
             "rolling_windows": [4, 16, 96],
         },
     }
     ```
     These mirror the constants in `features/wind.py:13-14`, `demand.py:13-14`, `solar.py:13-14`.

  3. `prepare_mlforecast_df(df, domain) -> pl.DataFrame` — Rename group col → `unique_id`, timestamp → `ds`, target → `y`. Keep exogenous columns. Drop raw/non-feature columns (qc_flag, status_code, etc.).

  4. `create_mlforecast(domain, config, horizons) -> MLForecast` — Instantiate the MLForecast object:
     ```python
     from mlforecast import MLForecast
     from mlforecast.lag_transforms import RollingMean, RollingStd
     import xgboost as xgb

     dcfg = DOMAIN_MLFORECAST[domain]
     lag_transforms = {}
     for w in dcfg["rolling_windows"]:
         # Attach rolling transforms to lag=1 (shift(1) then roll = same as our manual approach)
         lag_transforms.setdefault(1, []).extend([
             RollingMean(window_size=w),
             RollingStd(window_size=w),
         ])

     model = xgb.XGBRegressor(
         n_estimators=config.n_estimators,
         learning_rate=config.learning_rate,
         max_depth=config.max_depth,
         min_child_weight=config.min_child_weight,
         subsample=config.subsample,
         colsample_bytree=config.colsample_bytree,
         tree_method="hist",
     )

     fcst = MLForecast(
         models={"xgb": model},
         freq=dcfg["freq"],
         lags=dcfg["lags"],
         lag_transforms=lag_transforms,
     )
     return fcst
     ```

  5. `train_mlforecast(df, domain, config, horizons) -> MLForecast` — The main training function:
     ```python
     def train_mlforecast(
         df: pl.DataFrame,
         domain: str,
         config: MLForecastConfig | None = None,
         horizons: list[int] | None = None,
     ) -> MLForecast:
         if config is None:
             config = MLForecastConfig()
         if horizons is None:
             horizons = [1, 6, 12, 24, 48]

         fcst = create_mlforecast(domain, config, horizons)

         if config.strategy == "sparse_direct":
             fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", horizons=horizons)
         elif config.strategy == "direct":
             fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", max_horizon=max(horizons))
         else:  # recursive
             fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y")

         return fcst
     ```

  6. `predict_mlforecast(fcst, h, X_df=None) -> pl.DataFrame` — Predict wrapper.

  7. `cross_validate_mlforecast(df, domain, config, horizons, n_windows) -> pl.DataFrame` — CV wrapper using `fcst.cross_validation()`.

- **PATTERN**: Config pattern from `xgboost_model.py:14-24`, logging pattern from `train.py:185-243`
- **IMPORTS**: `mlforecast.MLForecast`, `mlforecast.lag_transforms.RollingMean`, `mlforecast.lag_transforms.RollingStd`, `xgboost as xgb`, `polars as pl`, `pydantic.BaseModel`, `mlflow`, `logging`
- **GOTCHA**: lag_transforms API verified: `RollingMean(window_size=w, min_samples=None)`. Import: `from mlforecast.lag_transforms import RollingMean, RollingStd`.
- **GOTCHA**: Polars `ds` column must be `pl.Datetime` type, not string. Our `timestamp_utc` is already Datetime — just rename.
- **GOTCHA**: freq format for Polars must use Polars duration strings: `'10m'` not `'10min'`, `'15m'` not `'15min'`, `'1h'` works as-is.
- **GOTCHA**: When using `horizons=[1,6,12,24,48]` (sparse direct), `predict(h=max(horizons))` returns predictions only at the specified horizons, not all intermediate steps.
- **GOTCHA**: LightGBM + Polars requires `as_numpy=True` in `fit()`. XGBoost works natively. Since we use XGBoost, no issue — but document this for future LightGBM support.
- **VALIDATE**: `uv run ruff check src/windcast/models/mlforecast_model.py && uv run pyright src/windcast/models/mlforecast_model.py`

### Task 6: UPDATE `src/windcast/models/__init__.py`

- **IMPLEMENT**: Export `MLForecastConfig`, `train_mlforecast`, `prepare_mlforecast_df`, `create_mlforecast`, `predict_mlforecast`, `cross_validate_mlforecast`
- **PATTERN**: Follow existing `__init__.py:1-25`
- **VALIDATE**: `uv run python -c "from windcast.models import MLForecastConfig, train_mlforecast"`

### Task 7: CREATE `scripts/train_mlforecast.py`

- **IMPLEMENT**: Training CLI script that mirrors `scripts/train.py` structure:
  1. Same argparse pattern: `--domain`, `--dataset`, `--feature-set`, `--experiment-name`, `--horizons`, `--turbine-id`
  2. Add `--strategy` flag: `{recursive, direct, sparse_direct}`, default `sparse_direct`
  3. Load processed Parquet (from `data/processed/`, NOT `data/features/`)
  4. Build exogenous features using `build_{domain}_exogenous()`
  5. Prepare mlforecast DataFrame using `prepare_mlforecast_df()`
  6. Temporal split (reuse same logic as train.py `_temporal_split()`)
  7. Train with `train_mlforecast()`
  8. Predict on validation set
  9. Compute metrics per horizon using existing `compute_metrics()`
  10. Log everything to MLflow (same nested run structure)

  Key difference from train.py: mlforecast returns predictions as a DataFrame with columns `[unique_id, ds, xgb]` (one row per horizon per series). Need to join back with actuals for evaluation.

- **PATTERN**: Mirror `scripts/train.py` line-for-line where possible (argparse, logging setup, MLflow setup, temporal split)
- **IMPORTS**: Everything from train.py + `windcast.models.mlforecast_model`, `windcast.features.exogenous`
- **GOTCHA**: After temporal split, pass only the train portion to `fcst.fit()`. For validation, use `fcst.predict()` — but mlforecast predicts from the END of the training data, not arbitrary points. For mid-dataset evaluation, use `fcst.cross_validation()` instead.
- **GOTCHA**: mlforecast `predict(h=48)` with `horizons=[1,6,12,24,48]` returns 5 rows per series (one per specified horizon). Map these back to our horizon-based evaluation.
- **VALIDATE**: `uv run ruff check scripts/train_mlforecast.py && uv run pyright scripts/train_mlforecast.py`

### Task 8: CREATE `tests/models/test_mlforecast_model.py`

- **IMPLEMENT**: Unit tests:
  1. `TestMLForecastConfig`: test defaults, test custom values
  2. `TestPrepareMLForecastDf`: test column renaming for each domain (wind/demand/solar)
  3. `TestCreateMLForecast`: test MLForecast object creation with correct lags/freq per domain
  4. `TestTrainMLForecast`: test end-to-end fit + predict with synthetic data (use sparse direct strategy with small horizons [1, 2])
  5. Test that predictions shape matches expected (n_series * n_horizons rows)

  Use synthetic data pattern from `test_xgboost_model.py:12-31` but adapted for mlforecast format:
  ```python
  def _make_mlforecast_data(n_rows=200, n_series=2):
      rng = np.random.default_rng(42)
      dates = pl.date_range(datetime(2020, 1, 1), periods=n_rows, interval="1h", eager=True)
      dfs = []
      for i in range(n_series):
          dfs.append(pl.DataFrame({
              "unique_id": [f"s{i}"] * n_rows,
              "ds": dates,
              "y": rng.standard_normal(n_rows).cumsum() + 100,
              "exog1": rng.standard_normal(n_rows),
          }))
      return pl.concat(dfs)
  ```

- **PATTERN**: Mirror `tests/models/test_xgboost_model.py` class structure
- **GOTCHA**: mlforecast needs enough data for lags + rolling windows. Use n_rows >= 200.
- **VALIDATE**: `uv run pytest tests/models/test_mlforecast_model.py -v`

### Task 9: CREATE `tests/features/test_exogenous.py`

- **IMPLEMENT**: Test exogenous feature builders:
  1. Test that wind exogenous builder produces expected columns (wind_dir_sin, wind_dir_cos, etc.) but NOT lag columns
  2. Test that demand exogenous builder produces calendar features, HDD/CDD but NOT load_mw_lag columns
  3. Test that solar exogenous builder produces clearsky_ratio but NOT power_kw_lag columns
  4. Test QC filtering (only QC_OK rows pass through)

- **PATTERN**: Mirror `tests/features/test_wind.py` patterns
- **VALIDATE**: `uv run pytest tests/features/test_exogenous.py -v`

---

## TESTING STRATEGY

### Unit Tests

- `tests/models/test_mlforecast_model.py` — Config, data prep, fit/predict, cross-validation
- `tests/features/test_exogenous.py` — Exogenous feature builders per domain

### Integration Tests (manual, via script)

- Run `scripts/train_mlforecast.py --domain wind` on actual Kelmarsh data
- Verify MLflow UI shows mlforecast experiment alongside XGBoost experiments
- Compare skill scores between XGBoost (manual) and mlforecast runs

### Edge Cases

- Single-series domain (demand: zone_id="ES") — verify mlforecast handles 1 series
- Very short horizons (h=1) with sparse_direct — verify predictions exist
- Missing exogenous columns (e.g., NWP not available) — verify graceful handling

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
```

**Expected**: All pass with exit code 0

### Level 2: Unit Tests

```bash
uv run pytest tests/models/test_mlforecast_model.py -v
uv run pytest tests/features/test_exogenous.py -v
```

### Level 3: Full Test Suite (no regressions)

```bash
uv run pytest tests/ -v
```

**Expected**: All 201+ tests pass (existing + new)

### Level 4: Manual Validation

```bash
# Wind domain end-to-end (requires Kelmarsh data in data/processed/)
uv run python scripts/train_mlforecast.py --domain wind --turbine-id kwf1

# Demand domain
uv run python scripts/train_mlforecast.py --domain demand

# Check MLflow
mlflow ui
# → Verify new experiment appears with metrics per horizon
```

---

## ACCEPTANCE CRITERIA

- [ ] `mlforecast>=1.0` installed and importable
- [ ] Exogenous feature builders produce correct columns (no lag/rolling contamination)
- [ ] `MLForecastConfig` follows Pydantic pattern from `XGBoostConfig`
- [ ] `prepare_mlforecast_df()` correctly maps all 3 domains (wind/demand/solar)
- [ ] `train_mlforecast()` fits and returns an MLForecast object
- [ ] Predictions have correct shape: `n_series * n_horizons` rows
- [ ] Metrics computed per horizon and logged to MLflow
- [ ] Existing XGBoost pipeline completely untouched (no regressions)
- [ ] All validation commands pass with zero errors
- [ ] New tests + existing 201 tests all pass
- [ ] `ruff check`, `pyright`, `pytest` all green

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order (1-9)
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully:
  - [ ] Level 1: ruff check, ruff format, pyright
  - [ ] Level 2: new test files pass
  - [ ] Level 3: full test suite passes (201+ tests)
  - [ ] Level 4: manual script testing on wind/demand
- [ ] No linting errors
- [ ] No type checking errors
- [ ] All acceptance criteria met

---

## NOTES

### Design Decisions

1. **Separate script (`train_mlforecast.py`) vs `--model` flag on `train.py`**: Chosen separate script because the data flow is fundamentally different (processed Parquet → mlforecast vs feature Parquet → manual XGBoost). Merging would add complexity for no benefit.

2. **Exogenous features split from lag/rolling**: mlforecast must own lags/rolling for correct recursive prediction. Domain-specific features (V³, clearsky_ratio, HDD/CDD) are exogenous and computed by us. This split is clean and mirrors the conceptual distinction: autoregressive vs physics-driven.

3. **Sparse direct as default strategy**: Our horizons `[1, 6, 12, 24, 48]` are sparse — training 48 models (full direct) is wasteful. Sparse direct trains exactly 5 models. This is the right default for energy forecasting.

4. **Keep existing pipeline**: The manual XGBoost pipeline is proven and well-tested. mlforecast is additive — users can compare both approaches in MLflow.

### Risk: mlforecast Polars compatibility — VERIFIED LOW RISK

Polars support confirmed since v0.11.0 (Nov 2023). `fit()` accepts `pl.DataFrame`, `predict()` returns `pl.DataFrame`. XGBoost works natively with Polars (LightGBM needs `as_numpy=True` — not relevant for us now). Only gotcha: freq must use Polars duration format (`'10m'`, `'15m'`, `'1h'`).

### Risk: lag_transforms API — VERIFIED, NO RISK

Exact imports and signatures confirmed from source:
```python
from mlforecast.lag_transforms import RollingMean, RollingStd
RollingMean(window_size=6, min_samples=None)
RollingStd(window_size=6, min_samples=None)
```
Also available: `RollingMin`, `RollingMax`, `RollingQuantile`, `ExponentiallyWeightedMean`, `ExpandingMean`, seasonal variants.

### Future: statsforecast baselines

This plan focuses on mlforecast only. A natural follow-up is adding `statsforecast` baselines (SeasonalNaive, AutoETS, AutoARIMA) — they use the same data format (unique_id, ds, y) so the `prepare_mlforecast_df()` function is reusable.
