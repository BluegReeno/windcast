# Feature: Phase 1.3 — Wind Feature Engineering + ML Pipeline

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types and models. Import from the right files.

## Feature Description

Build the complete wind ML pipeline: feature engineering (lag, rolling, cyclic, wind-specific), feature set registry (baseline/enriched/full), persistence benchmark, XGBoost training with MLflow tracking, evaluation with skill scores and regime analysis, and CLI scripts for each step. This transforms clean SCADA Parquet files into tracked ML experiments with skill score > 0 at all horizons.

## User Story

As an ML engineer
I want to run `build_features.py → train.py → evaluate.py` and see results in MLflow UI
So that I can iterate on feature sets and models with full reproducibility

## Problem Statement

Phase 1.2 produced clean, QC'd SCADA Parquet files per turbine. But these raw 15-column files cannot be fed directly to ML models — they need feature engineering (lags, rolling statistics, cyclic encoding, wind-specific transforms), temporal train/val/test splitting, a naive baseline to beat, and proper experiment tracking. Without this, there's no way to prove the framework works end-to-end.

## Solution Statement

Build 6 modules + 3 scripts:
1. **features/registry.py** — Declarative feature set registry (baseline/enriched/full), each set is a named list of feature column names
2. **features/wind.py** — Wind-specific feature builders: lags, rolling stats, cyclic encoding, V³, turbulence intensity, direction sectors
3. **models/persistence.py** — Naive persistence baseline (last known power = forecast)
4. **models/xgboost_model.py** — XGBoost trainer with MLflow logging, early stopping, per-horizon models
5. **models/evaluation.py** — Metrics (MAE, RMSE, MAPE, skill score, bias), regime analysis, custom metric support
6. **tracking/mlflow_utils.py** — MLflow setup helpers (experiment creation, artifact logging, nested runs)
7. **scripts/build_features.py** — CLI: load Parquet → engineer features → save feature Parquet
8. **scripts/train.py** — CLI: load features → temporal split → train per horizon → MLflow logging
9. **scripts/evaluate.py** — CLI: load model + test data → evaluate → MLflow artifacts

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: High
**Primary Systems Affected**: `src/windcast/features/`, `src/windcast/models/`, `src/windcast/tracking/`, `scripts/`
**Dependencies**: xgboost, scikit-learn, mlflow, polars (all already in pyproject.toml)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `src/windcast/config.py` (full file) — Settings with `features_dir`, `forecast_horizons`, `mlflow_tracking_uri`, `train_years`/`val_years`/`test_years`
- `src/windcast/data/schema.py` (full file) — SCADA_SCHEMA (15 cols), SIGNAL_COLUMNS, QC_OK/QC_SUSPECT/QC_BAD constants
- `src/windcast/data/qc.py` (full file) — Pipeline composition pattern: chain functions, each takes/returns DataFrame. Use as model for feature pipeline
- `src/windcast/data/kelmarsh.py` (full file) — Logging pattern, error handling pattern, private helpers with `_` prefix
- `scripts/ingest_kelmarsh.py` (full file) — Script pattern: argparse, logging setup, get_settings(), step-by-step with logging
- `tests/data/test_qc.py` (full file) — Test pattern: `_make_scada_df()` factory, class-based grouping, specific assertions
- `tests/data/test_open_meteo.py` (full file) — Mock pattern: MagicMock for external deps
- `.claude/docs/ml-pipeline-patterns.md` (full file) — **CRITICAL**: XGBoost, MLflow, Polars interop patterns, gotchas

### New Files to Create

- `src/windcast/features/registry.py` — Feature set definitions (baseline/enriched/full)
- `src/windcast/features/wind.py` — Wind feature engineering functions
- `src/windcast/features/__init__.py` — Update with exports
- `src/windcast/models/persistence.py` — Persistence baseline model
- `src/windcast/models/xgboost_model.py` — XGBoost training wrapper
- `src/windcast/models/evaluation.py` — Metrics and evaluation functions
- `src/windcast/models/__init__.py` — Update with exports
- `src/windcast/tracking/__init__.py` — Package init
- `src/windcast/tracking/mlflow_utils.py` — MLflow helpers
- `scripts/build_features.py` — Feature building CLI
- `scripts/train.py` — Training CLI
- `scripts/evaluate.py` — Evaluation CLI
- `tests/features/__init__.py` — Test package
- `tests/features/test_registry.py` — Registry tests
- `tests/features/test_wind.py` — Wind feature tests
- `tests/models/__init__.py` — Test package
- `tests/models/test_persistence.py` — Persistence tests
- `tests/models/test_evaluation.py` — Evaluation/metrics tests
- `tests/models/test_xgboost_model.py` — XGBoost wrapper tests

### Relevant Documentation

Read `.claude/docs/ml-pipeline-patterns.md` before implementing — contains tested code patterns for:
- XGBoost quantile regression with early stopping (section 1)
- MLflow nested runs pattern (section 2.3)
- Polars lag/rolling features with shift(1) gotcha (section 3.4)
- Cyclic encoding in Polars (section 3.5)
- scikit-learn metrics — use `root_mean_squared_error` NOT `mean_squared_error(squared=False)` (section 4)
- Persistence baseline (section 5)
- Complete training script skeleton (section 6)

### Patterns to Follow

**Naming Conventions:**
- Files: `snake_case.py`
- Functions: `snake_case` (public), `_snake_case` (private helpers)
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

**Module Structure (mirror qc.py):**
- Module-level `logger = logging.getLogger(__name__)`
- Single public orchestrator function (e.g., `build_wind_features()`)
- Private helper functions for each transform (e.g., `_add_lag_features()`)
- Config via optional parameter with default (e.g., `settings: WindCastSettings | None = None`)

**Error Handling:**
- `ValueError` for invalid inputs with descriptive message
- `FileNotFoundError` for missing data files
- Assertions for internal invariants only
- Log warnings for non-fatal issues, continue gracefully

**Logging Pattern:**
```python
logger = logging.getLogger(__name__)
logger.info("Building features for %s (%d rows)", turbine_id, len(df))
logger.warning("Missing column %s, skipping feature", col_name)
```

**Test Pattern:**
- Factory function `_make_feature_df(n_rows=100, **overrides)` returning a Polars DataFrame
- Class-based test grouping: `class TestLagFeatures:`, `class TestCyclicFeatures:`
- Specific assertions: exact column names, dtypes, null count, value ranges

**DataFrame Flow:**
- Input: `pl.DataFrame` with SCADA_SCHEMA columns + `qc_flag` filtering
- Output: `pl.DataFrame` with original columns + feature columns
- All transforms via `.with_columns()` (non-destructive)
- Temporal operations use `.shift(1)` before `.rolling_mean()` to prevent look-ahead

---

## IMPLEMENTATION PLAN

### Phase 1: Feature Engineering Foundation

Build the feature set registry and wind-specific feature builders. These are pure Polars transforms with no external dependencies.

### Phase 2: Models & Evaluation

Build persistence baseline, XGBoost wrapper, and evaluation metrics. These depend on features being defined.

### Phase 3: MLflow Tracking

Build MLflow utilities for experiment setup, logging, and artifact management.

### Phase 4: CLI Scripts

Build the 3 CLI scripts that orchestrate the pipeline: build_features → train → evaluate.

### Phase 5: Testing & Validation

Tests for each module, following existing test patterns.

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/windcast/features/registry.py`

Feature set registry — declarative definitions of which columns belong to each set.

- **IMPLEMENT**: Define feature sets as named tuples or dataclasses:
  - `WIND_BASELINE`: `["wind_speed_ms", "wind_dir_sin", "wind_dir_cos", "active_power_kw_lag1", "active_power_kw_lag2", "active_power_kw_lag3", "active_power_kw_lag6", "active_power_kw_roll_mean_6", "active_power_kw_roll_mean_12", "active_power_kw_roll_mean_24", "active_power_kw_roll_std_6"]`
  - `WIND_ENRICHED`: baseline + `["wind_speed_cubed", "turbulence_intensity", "wind_dir_sector", "hour_sin", "hour_cos"]`
  - `WIND_FULL`: enriched + NWP columns `["nwp_wind_speed_100m", "nwp_wind_direction_100m", "nwp_temperature_2m", "month_sin", "month_cos", "dow_sin", "dow_cos"]`
- **IMPLEMENT**: `FeatureSet` dataclass with `name: str`, `columns: list[str]`, `description: str`
- **IMPLEMENT**: `FEATURE_REGISTRY: dict[str, FeatureSet]` with keys `"wind_baseline"`, `"wind_enriched"`, `"wind_full"`
- **IMPLEMENT**: `get_feature_set(name: str) -> FeatureSet` lookup function
- **IMPLEMENT**: `list_feature_sets() -> list[str]` for CLI help
- **PATTERN**: Mirror `DATASETS` dict pattern in `config.py`
- **IMPORTS**: `from dataclasses import dataclass`
- **GOTCHA**: Column names must exactly match what `wind.py` produces
- **VALIDATE**: `uv run python -c "from windcast.features.registry import get_feature_set; fs = get_feature_set('wind_baseline'); print(fs.columns)"`

### Task 2: CREATE `src/windcast/features/wind.py`

Wind-specific feature engineering functions.

- **IMPLEMENT**: `build_wind_features(df: pl.DataFrame, feature_set: str = "wind_baseline") -> pl.DataFrame`
  - Public orchestrator that chains private helpers based on requested feature set
  - Filter to `qc_flag == QC_OK` rows only (drop suspect/bad before features)
  - Sort by `timestamp_utc` per turbine before computing lags
- **IMPLEMENT**: `_add_lag_features(df, col, lags) -> pl.DataFrame`
  - Lags: [1, 2, 3, 6, 12, 24] steps (10-min intervals)
  - Pattern: `pl.col(col).shift(lag).over("turbine_id").alias(f"{col}_lag{lag}")`
  - **GOTCHA**: Must use `.over("turbine_id")` for per-turbine lags
- **IMPLEMENT**: `_add_rolling_features(df, col, windows) -> pl.DataFrame`
  - Windows: [6, 12, 24] (= 1h, 2h, 4h at 10-min resolution)
  - Pattern: `pl.col(col).shift(1).rolling_mean(window_size=w).over("turbine_id")`
  - **GOTCHA**: `.shift(1)` BEFORE `.rolling_mean()` to prevent look-ahead leakage
  - Add both `_roll_mean_` and `_roll_std_` variants
- **IMPLEMENT**: `_add_cyclic_features(df) -> pl.DataFrame`
  - Hour: sin/cos with period 24
  - Month: sin/cos with period 12
  - Day of week: sin/cos with period 7
  - Wind direction: sin/cos with period 360 (degrees to radians)
  - Pattern: `(pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24)).sin()`
- **IMPLEMENT**: `_add_wind_specific_features(df) -> pl.DataFrame`
  - `wind_speed_cubed`: `pl.col("wind_speed_ms").pow(3)` — captures cubic power relationship
  - `turbulence_intensity`: `pl.col("wind_speed_ms").shift(1).rolling_std(window_size=6).over("turbine_id") / pl.col("wind_speed_ms")` — TI = sigma_v / mean_v
  - `wind_dir_sector`: `(pl.col("wind_direction_deg") / 30).cast(pl.Int32)` — 12 sectors of 30°
- **IMPORTS**: `import math`, `import polars as pl`, `from windcast.data.schema import QC_OK`, `from windcast.features.registry import get_feature_set`
- **VALIDATE**: `uv run ruff check src/windcast/features/wind.py && uv run pyright src/windcast/features/wind.py`

### Task 3: UPDATE `src/windcast/features/__init__.py`

- **IMPLEMENT**: Export public API:
  ```python
  from windcast.features.registry import FEATURE_REGISTRY, FeatureSet, get_feature_set, list_feature_sets
  from windcast.features.wind import build_wind_features

  __all__ = ["FEATURE_REGISTRY", "FeatureSet", "build_wind_features", "get_feature_set", "list_feature_sets"]
  ```
- **VALIDATE**: `uv run python -c "from windcast.features import build_wind_features, get_feature_set"`

### Task 4: CREATE `src/windcast/models/persistence.py`

Naive persistence baseline.

- **IMPLEMENT**: `persistence_forecast(y_true: np.ndarray, y_lag1: np.ndarray) -> np.ndarray`
  - Simply returns `y_lag1` — the last known power value is the forecast
  - For horizon h, the "lag1" in feature matrix already represents t-1, so persistence for h>1 is still "use lag1"
  - This is intentionally simple — persistence baseline doesn't need to be horizon-aware in a direct forecasting setup where we train separate models per horizon
- **IMPLEMENT**: `compute_persistence_metrics(y_true: np.ndarray, y_lag1: np.ndarray) -> dict[str, float]`
  - Returns dict with `mae`, `rmse`, `bias` for the persistence baseline
- **IMPORTS**: `import numpy as np`, `from sklearn.metrics import mean_absolute_error, root_mean_squared_error`
- **GOTCHA**: Persistence is only valid when `y_lag1` corresponds to `active_power_kw_lag1` in the feature matrix
- **VALIDATE**: `uv run python -c "from windcast.models.persistence import persistence_forecast; import numpy as np; print(persistence_forecast(np.array([1,2,3]), np.array([0,1,2])))"`

### Task 5: CREATE `src/windcast/models/evaluation.py`

Evaluation metrics and regime analysis.

- **IMPLEMENT**: `compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_persistence: np.ndarray | None = None) -> dict[str, float]`
  - Returns: `mae`, `rmse`, `bias` (= mean(y_pred - y_true))
  - If `y_persistence` provided: also `skill_score` (1 - RMSE_model / RMSE_persistence)
  - Optional `mape` — only computed when no zeros in y_true (log warning if skipped)
- **IMPLEMENT**: `compute_skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_persistence: np.ndarray) -> float`
  - `1 - RMSE_model / RMSE_persistence`
  - Handle edge case: if RMSE_persistence == 0, return 1.0 if RMSE_model == 0 else -inf
- **IMPLEMENT**: `regime_analysis(df: pl.DataFrame, y_true_col: str, y_pred_col: str, wind_speed_col: str = "wind_speed_ms") -> dict[str, dict[str, float]]`
  - Split into regimes: low wind (<5 m/s), medium (5-12 m/s), high (>12 m/s)
  - Compute MAE, RMSE per regime
  - Return `{"low": {"mae": ..., "rmse": ...}, "medium": {...}, "high": {...}}`
- **IMPLEMENT**: Type alias `CustomMetric = Callable[[np.ndarray, np.ndarray], float]` for pluggable metrics
- **IMPLEMENT**: `evaluate_with_custom_metrics(y_true, y_pred, custom_metrics: dict[str, CustomMetric]) -> dict[str, float]`
- **IMPORTS**: `from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error`
- **GOTCHA**: Use `root_mean_squared_error` (sklearn >=1.4), NOT `mean_squared_error(squared=False)` (removed in 1.6)
- **GOTCHA**: MAPE is undefined with zeros in y_true — check before computing, skip with warning
- **VALIDATE**: `uv run ruff check src/windcast/models/evaluation.py && uv run pyright src/windcast/models/evaluation.py`

### Task 6: CREATE `src/windcast/models/xgboost_model.py`

XGBoost training wrapper with MLflow integration.

- **IMPLEMENT**: `XGBoostConfig` — Pydantic BaseModel for hyperparameters:
  ```python
  class XGBoostConfig(BaseModel):
      objective: str = "reg:squarederror"
      n_estimators: int = 500
      learning_rate: float = 0.05
      max_depth: int = 6
      min_child_weight: int = 10
      subsample: float = 0.8
      colsample_bytree: float = 0.8
      tree_method: str = "hist"
      early_stopping_rounds: int = 50
  ```
- **IMPLEMENT**: `train_xgboost(X_train: pl.DataFrame, y_train: pl.Series, X_val: pl.DataFrame, y_val: pl.Series, config: XGBoostConfig | None = None) -> xgb.XGBRegressor`
  - Creates XGBRegressor from config
  - Fits with eval_set and early stopping
  - Logs to MLflow if active run exists
  - Returns fitted model
  - **GOTCHA**: Pass Polars DataFrames directly to XGBoost (>=2.0 supports it) — feature names auto-inferred
- **IMPLEMENT**: `train_multi_horizon(X_train, y_trains: dict[int, pl.Series], X_val, y_vals: dict[int, pl.Series], config=None) -> dict[int, xgb.XGBRegressor]`
  - Trains one model per horizon
  - Uses MLflow nested runs (one child per horizon)
  - Returns dict mapping horizon → model
- **IMPORTS**: `import xgboost as xgb`, `import mlflow`, `from pydantic import BaseModel`
- **GOTCHA**: `nested=True` in `mlflow.start_run()` for child runs
- **VALIDATE**: `uv run ruff check src/windcast/models/xgboost_model.py && uv run pyright src/windcast/models/xgboost_model.py`

### Task 7: CREATE `src/windcast/tracking/__init__.py` and `src/windcast/tracking/mlflow_utils.py`

MLflow setup utilities.

- **CREATE**: `src/windcast/tracking/__init__.py` with exports
- **IMPLEMENT** in `mlflow_utils.py`:
  - `setup_mlflow(tracking_uri: str = "file:./mlruns", experiment_name: str | None = None) -> None`
    - Calls `mlflow.set_tracking_uri()` and optionally `mlflow.set_experiment()`
  - `log_feature_set(feature_set_name: str, feature_columns: list[str]) -> None`
    - Logs feature set as MLflow param + JSON artifact
  - `log_evaluation_results(metrics: dict[str, float], horizon: int | None = None) -> None`
    - Logs metrics dict, optionally prefixed by horizon
  - `log_dataframe_artifact(df: pl.DataFrame, name: str) -> None`
    - Saves DataFrame as CSV to temp file, logs as MLflow artifact
- **IMPORTS**: `import mlflow`, `import polars as pl`, `import tempfile`
- **VALIDATE**: `uv run ruff check src/windcast/tracking/ && uv run pyright src/windcast/tracking/`

### Task 8: UPDATE `src/windcast/models/__init__.py`

- **IMPLEMENT**: Export public API from persistence, evaluation, xgboost_model
- **VALIDATE**: `uv run python -c "from windcast.models import compute_metrics, train_xgboost"`

### Task 9: CREATE `scripts/build_features.py`

Feature building CLI script.

- **IMPLEMENT**: argparse CLI with flags:
  - `--input-dir` (default: `data/processed/`)
  - `--output-dir` (default: `data/features/`)
  - `--feature-set` (default: `"wind_baseline"`, choices from registry)
  - `--turbine-id` (optional, default: all turbines)
- **IMPLEMENT**: Main flow:
  1. Glob `input_dir` for `kelmarsh_*.parquet` files
  2. For each file: `pl.read_parquet()` → `build_wind_features(df, feature_set)` → drop nulls → write Parquet
  3. Log summary: rows in, rows out, features added
- **PATTERN**: Mirror `scripts/ingest_kelmarsh.py` structure (argparse, logging.basicConfig, get_settings)
- **VALIDATE**: `uv run ruff check scripts/build_features.py`

### Task 10: CREATE `scripts/train.py`

Training CLI script with MLflow tracking.

- **IMPLEMENT**: argparse CLI with flags:
  - `--features-dir` (default: `data/features/`)
  - `--feature-set` (default: `"wind_baseline"`)
  - `--turbine-id` (default: `"kwf1"` — single turbine for now)
  - `--experiment-name` (default: `"windcast-kelmarsh"`)
  - `--horizons` (default from settings: `[1, 6, 12, 24, 48]`)
- **IMPLEMENT**: Main flow:
  1. Load feature Parquet for specified turbine
  2. Temporal split: use `train_years`/`val_years`/`test_years` from settings
     - Compute split dates from data timestamp range
     - train = first `train_years`, val = next `val_years`, test = last `test_years`
  3. For each horizon h:
     - Build target: `pl.col("active_power_kw").shift(-h)` then drop nulls
     - Extract X (feature columns) and y (target)
  4. Setup MLflow experiment
  5. Parent run: log dataset info, feature set, hyperparams
  6. For each horizon: child run with `nested=True`, train XGBoost, log metrics
  7. Log persistence baseline metrics for comparison
  8. Save models as MLflow artifacts
- **PATTERN**: Mirror `scripts/ingest_kelmarsh.py` for CLI structure
- **GOTCHA**: Target construction — shift(-h) creates the future target, then drop nulls at the end of the series
- **GOTCHA**: Temporal split must NOT shuffle — use date-based filtering
- **VALIDATE**: `uv run ruff check scripts/train.py`

### Task 11: CREATE `scripts/evaluate.py`

Evaluation CLI script.

- **IMPLEMENT**: argparse CLI with flags:
  - `--features-dir`, `--turbine-id`, `--feature-set`, `--experiment-name`
  - `--run-id` (optional — evaluate latest run if not specified)
- **IMPLEMENT**: Main flow:
  1. Load test data (from temporal split)
  2. Load trained models from MLflow (by run ID or latest)
  3. For each horizon: predict on test set, compute metrics, compute skill score vs persistence
  4. Regime analysis (low/med/high wind)
  5. Log all results to MLflow (as child run of training run, or new run)
  6. Print summary table to stdout
- **VALIDATE**: `uv run ruff check scripts/evaluate.py`

### Task 12: CREATE tests

- **CREATE** `tests/features/__init__.py`
- **CREATE** `tests/features/test_registry.py`:
  - `test_get_feature_set_baseline` — returns correct columns
  - `test_get_feature_set_enriched_extends_baseline` — enriched is superset of baseline
  - `test_get_feature_set_unknown_raises` — ValueError for unknown name
  - `test_list_feature_sets` — returns all registered names
- **CREATE** `tests/features/test_wind.py`:
  - Factory: `_make_wind_df(n_rows=200)` — creates DataFrame with SCADA columns + realistic values
  - `class TestLagFeatures:` — verify lag columns exist, values are shifted correctly, nulls at start
  - `class TestRollingFeatures:` — verify rolling columns, no look-ahead (shift(1) applied)
  - `class TestCyclicFeatures:` — verify sin/cos columns, values in [-1, 1] range
  - `class TestWindSpecificFeatures:` — verify V³, turbulence intensity, direction sectors
  - `class TestBuildWindFeatures:` — integration: full pipeline produces correct output shape and columns
- **CREATE** `tests/models/__init__.py`
- **CREATE** `tests/models/test_persistence.py`:
  - `test_persistence_returns_lag1` — output equals input lag
  - `test_persistence_metrics_computed` — metrics dict has expected keys
- **CREATE** `tests/models/test_evaluation.py`:
  - `test_compute_metrics_basic` — perfect prediction gives MAE=0, RMSE=0
  - `test_skill_score_perfect` — skill=1.0 for perfect model
  - `test_skill_score_same_as_persistence` — skill=0.0
  - `test_skill_score_worse_than_persistence` — skill < 0
  - `test_mape_skipped_with_zeros` — no error, warning logged
  - `test_regime_analysis_three_regimes` — returns low/medium/high keys
  - `test_custom_metrics` — pluggable metric function works
- **CREATE** `tests/models/test_xgboost_model.py`:
  - `test_xgboost_config_defaults` — default config valid
  - `test_train_xgboost_returns_fitted_model` — smoke test with tiny data
  - Mock MLflow calls to avoid actual tracking in tests
- **VALIDATE**: `uv run pytest tests/ -v`

### Task 13: Final validation

- **VALIDATE**: `uv run ruff check src/ tests/ scripts/`
- **VALIDATE**: `uv run ruff format --check src/ tests/ scripts/`
- **VALIDATE**: `uv run pyright src/`
- **VALIDATE**: `uv run pytest tests/ -v`

---

## TESTING STRATEGY

### Unit Tests

- **Features**: Test each transform in isolation with synthetic data. Verify column names, dtypes, value ranges, null positions.
- **Evaluation**: Test metrics with known inputs (perfect prediction, known error values). Test edge cases (zeros in y_true for MAPE).
- **Persistence**: Test that output equals lag1 input. Test metrics computation.
- **Registry**: Test lookup, unknown key error, list.

### Integration Tests

- **build_wind_features()**: Full pipeline from SCADA DataFrame to feature DataFrame. Verify all expected columns present, no look-ahead leakage.
- **train_xgboost()**: Smoke test with tiny synthetic data (50 rows). Verify model returns predictions, no errors.

### Edge Cases

- Empty DataFrame input to feature builder
- Single turbine vs multiple turbines in feature DataFrame
- Zero wind speed (turbulence intensity → division by zero)
- All QC_BAD rows filtered out → empty result
- Horizon larger than available data → graceful error

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
```

### Level 2: Unit Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/features/ -v
uv run pytest tests/models/ -v
```

### Level 3: Integration Test (manual)

```bash
# Requires Kelmarsh data ingested first
uv run python scripts/ingest_kelmarsh.py
uv run python scripts/build_features.py --feature-set wind_baseline
uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_baseline
uv run python scripts/evaluate.py --turbine-id kwf1
mlflow ui  # Check results at http://localhost:5000
```

---

## ACCEPTANCE CRITERIA

- [ ] `build_wind_features()` produces correct columns for all 3 feature sets
- [ ] No look-ahead leakage in lag/rolling features (shift(1) before rolling)
- [ ] Persistence baseline computes correct RMSE
- [ ] XGBoost trains with early stopping, logs to MLflow
- [ ] Skill score > 0 at short horizons (1-6 steps) for Kelmarsh data
- [ ] MLflow UI shows parent run with child runs per horizon
- [ ] Feature importance logged as MLflow artifact
- [ ] Regime analysis produces metrics for low/medium/high wind
- [ ] Custom metric support works (callable interface)
- [ ] All validation commands pass with zero errors
- [ ] No regressions in existing 50 tests

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order (1-13)
- [ ] Level 1 validation: ruff check, ruff format, pyright — all pass
- [ ] Level 2 validation: pytest — all pass (old + new tests)
- [ ] Level 3 validation: end-to-end pipeline runs, MLflow shows results
- [ ] All acceptance criteria met

---

## NOTES

### Design Decisions

1. **Feature set as registry, not config file** — Feature sets are code (lists of column names), not YAML/JSON. This ensures type safety and IDE autocomplete. Easy to add new sets.

2. **Separate models per horizon** — Direct multi-step forecasting (one model per horizon) is simpler and often better than recursive forecasting for wind power. Each model can learn horizon-specific patterns.

3. **Persistence baseline is trivially simple** — It just returns lag1. This is correct for direct forecasting: at each horizon, the naive forecast is "power stays the same as last observation." The skill score measures how much better XGBoost is.

4. **QC filtering before features** — Filter to `qc_flag == QC_OK` before computing lags/rolling. This prevents contaminated data from leaking into features. Tradeoff: loses some rows at QC boundaries, but data integrity is more important.

5. **Per-turbine operations with `.over("turbine_id")`** — All lags and rolling stats must be computed per turbine. Cross-turbine contamination would be a critical bug.

6. **MLflow nested runs** — Parent run per experiment (logs dataset info, feature set), child runs per horizon (logs per-horizon metrics and model). This gives clean comparison in MLflow UI.

### Key Gotchas

- `.shift(1)` before `.rolling_mean()` — **CRITICAL**: prevents look-ahead leakage
- `.over("turbine_id")` on all temporal operations — prevents cross-turbine contamination
- `root_mean_squared_error` not `mean_squared_error(squared=False)` — the latter is removed in sklearn 1.6
- `nested=True` in MLflow child runs — mandatory in MLflow >=2.x
- Turbulence intensity = std/mean — division by zero when wind_speed=0, handle with `.fill_nan(0)`
- MAPE undefined with zero y_true — skip and log warning
- XGBoost accepts Polars DataFrames directly (>=2.0) — no `.to_numpy()` needed
