# Feature: AutoGluon-Tabular as 3rd ML Backend

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to: disabling MLflow autolog before AutoGluon fit, Polars→pandas boundary, score sign convention (negated in leaderboard).

## Feature Description

Add AutoGluon-Tabular as a 3rd ML backend alongside XGBoost (manual) and mlforecast (Nixtla). AutoGluon automatically trains and stacks multiple models (LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees → WeightedEnsemble). This is WN's own tool — showing it plugs into EnerCast in one file is a key demo argument.

## User Story

As a WN evaluator,
I want to see AutoGluon-Tabular plugged into the same pipeline with zero core changes,
So that I believe the framework is truly backend-agnostic and supports our existing tools.

## Problem Statement

EnerCast currently has 2 ML backends (XGBoost manual, mlforecast). WN uses AutoGluon for bagging/stacking. Adding it as a 3rd backend proves pluggability and uses WN's own stack.

## Solution Statement

Create `src/windcast/models/autogluon_model.py` (thin wrapper) + `scripts/train_autogluon.py` (training script). Uses the same feature Parquets as XGBoost. Logs to MLflow manually (AutoGluon is incompatible with MLflow autolog). Leaderboard + best model metrics visible in MLflow UI.

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: `models/`, `scripts/`, `pyproject.toml`
**Dependencies**: `autogluon.tabular[all]`

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `src/windcast/models/xgboost_model.py` (full file) — Pattern: Pydantic config + train function + train_multi_horizon. Mirror the config/train pattern.
- `scripts/train.py` (full file) — Pattern: argparse, temporal split, per-horizon loop with nested MLflow runs, parent description, summary metrics bubble-up. **This is the primary template.**
- `scripts/train_mlforecast.py` (full file) — Pattern: alternative backend script with manual MLflow logging (no autolog). Mirror this pattern for AutoGluon.
- `src/windcast/models/__init__.py` — Must add AutoGluon exports here.
- `src/windcast/models/evaluation.py` — `compute_metrics()` and `compute_persistence_metrics()` — reuse for evaluation.
- `src/windcast/features/registry.py` — Feature sets: AutoGluon uses the SAME feature sets as XGBoost (wind_baseline, wind_enriched, wind_full).
- `src/windcast/tracking/mlflow_utils.py` — `setup_mlflow()`, `log_evaluation_results()` — reuse.
- `tests/models/test_xgboost_model.py` — Test pattern: `_make_regression_data()` helper, mock mlflow, test config + train.
- `.claude/docs/autogluon-patterns.md` — API reference, gotchas, MLflow wrapper patterns.
- `.claude/docs/autogluon-vs-manual-tradeoffs.md` — Tradeoff analysis, recommendation to use as benchmark lane.

### New Files to Create

- `src/windcast/models/autogluon_model.py` — AutoGluon-Tabular trainer wrapper
- `scripts/train_autogluon.py` — Training script (same interface as train.py)
- `tests/models/test_autogluon_model.py` — Unit tests

### Relevant Documentation

**AutoGluon-Tabular API (v1.5.0):**
- TabularPredictor constructor: `TabularPredictor(label=..., problem_type="regression", eval_metric="mean_absolute_error", path=...)`
- fit: `predictor.fit(train_data, presets="best_quality", time_limit=300, excluded_model_types=["NN_TORCH", "FASTAI", "TABPFNV2", "TabICL", "MITRA"])`
- predict: `predictor.predict(test_data)` → pd.Series
- leaderboard: `predictor.leaderboard(data=test_df, silent=True)` → pd.DataFrame with model, score_val, score_test, fit_time, pred_time_val, stack_level
- feature_importance: `predictor.feature_importance(data=val_df)` → pd.DataFrame

**Critical gotchas:**
- **MUST disable MLflow autolog before fit**: `mlflow.autolog(disable=True)` — AutoGluon's sklearn internals break with autolog monkey-patching (PicklingError)
- **Re-enable autolog after fit** if other code needs it
- **Score sign convention**: leaderboard scores are higher-is-better, so MAE shows as NEGATIVE (e.g., -115.3 means MAE=115.3). Negate when logging to MLflow.
- **Polars→pandas boundary**: call `.to_pandas()` before passing to AutoGluon. Keep Polars for all upstream processing.
- **Exclude GPU models on macOS**: `excluded_model_types=["NN_TORCH", "FASTAI", "TABPFNV2", "TabICL", "MITRA"]`

### Patterns to Follow

**Config pattern** (from xgboost_model.py):
```python
class AutoGluonConfig(BaseModel):
    presets: str = "best_quality"
    time_limit: int = 300  # seconds per horizon
    excluded_model_types: list[str] = ["NN_TORCH", "FASTAI", "TABPFNV2", "TabICL", "MITRA"]
    eval_metric: str = "mean_absolute_error"
```

**Training function pattern** (from xgboost_model.py):
```python
def train_autogluon(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_val: pl.DataFrame,
    y_val: pl.Series,
    config: AutoGluonConfig | None = None,
) -> TabularPredictor:
```

**MLflow logging pattern** (from train.py):
- Parent run: `{run_label}-autogluon-{feature_set}`
- Tags: `enercast.backend = "autogluon"`
- Per-horizon child runs with metrics
- Parent description with results table
- Summary metrics bubbled up to parent

---

## IMPLEMENTATION PLAN

### Phase 1: Add Dependency

Add `autogluon.tabular[all]` to pyproject.toml.

### Phase 2: Model Wrapper

Create `autogluon_model.py` with:
- `AutoGluonConfig` (Pydantic BaseModel)
- `train_autogluon()` — trains TabularPredictor for a single horizon, returns predictor
- Handles Polars→pandas conversion at the boundary
- Handles autolog disable/re-enable

### Phase 3: Training Script

Create `scripts/train_autogluon.py` mirroring `train.py`:
- Same argparse interface (--domain, --feature-set, --turbine-id, --horizons, etc.)
- Same temporal split logic
- Same per-horizon nested MLflow runs
- Plus: log leaderboard as artifact, log best model name, log feature importance
- Parent run description with results table

### Phase 4: Tests

Unit tests for config and train function with small synthetic data.

### Phase 5: Update Exports

Add to `models/__init__.py`.

---

## STEP-BY-STEP TASKS

### Task 1: UPDATE `pyproject.toml`

- **IMPLEMENT**: Add `"autogluon.tabular[all]>=1.2"` to dependencies list
- **VALIDATE**: `uv sync` completes without errors

### Task 2: CREATE `src/windcast/models/autogluon_model.py`

- **IMPLEMENT**:
  ```python
  """AutoGluon-Tabular training wrapper with MLflow-safe autolog handling."""
  
  import logging
  import tempfile
  from pathlib import Path
  
  import mlflow
  import polars as pl
  from pydantic import BaseModel
  
  logger = logging.getLogger(__name__)
  
  
  class AutoGluonConfig(BaseModel):
      """Configuration for AutoGluon-Tabular training."""
      presets: str = "best_quality"
      time_limit: int = 300
      excluded_model_types: list[str] = [
          "NN_TORCH", "FASTAI", "TABPFNV2", "TabICL", "MITRA",
      ]
      eval_metric: str = "mean_absolute_error"
  
  
  def train_autogluon(
      X_train: pl.DataFrame,
      y_train: pl.Series,
      X_val: pl.DataFrame,
      y_val: pl.Series,
      config: AutoGluonConfig | None = None,
      ag_path: Path | None = None,
  ) -> "TabularPredictor":
      """Train AutoGluon-Tabular on a single regression task.
      
      Disables MLflow autolog during fit (AutoGluon is incompatible).
      Converts Polars to pandas at the boundary.
      
      Args:
          X_train, y_train: Training data (Polars).
          X_val, y_val: Validation data (Polars).
          config: AutoGluon config. Uses defaults if None.
          ag_path: Directory for AutoGluon artifacts. Uses tempdir if None.
      
      Returns:
          Fitted TabularPredictor.
      """
      from autogluon.tabular import TabularPredictor
      
      if config is None:
          config = AutoGluonConfig()
      
      # Polars → pandas at the AutoGluon boundary
      label = y_train.name
      train_pd = X_train.to_pandas()
      train_pd[label] = y_train.to_pandas()
      val_pd = X_val.to_pandas()
      val_pd[label] = y_val.to_pandas()
      
      # CRITICAL: Disable MLflow autolog — AutoGluon breaks with sklearn monkey-patching
      mlflow.autolog(disable=True)
      
      path = str(ag_path) if ag_path else tempfile.mkdtemp(prefix="ag_")
      
      predictor = TabularPredictor(
          label=label,
          problem_type="regression",
          eval_metric=config.eval_metric,
          path=path,
          verbosity=1,
      )
      
      predictor.fit(
          train_data=train_pd,
          tuning_data=val_pd,
          presets=config.presets,
          time_limit=config.time_limit,
          excluded_model_types=config.excluded_model_types,
      )
      
      # Re-enable autolog for other code
      mlflow.autolog(disable=False)
      
      # Log summary
      lb = predictor.leaderboard(data=val_pd, silent=True)
      best_model = lb.iloc[0]["model"]
      best_score = -lb.iloc[0]["score_val"]  # Negate: AG uses higher-is-better
      n_models = len(lb)
      
      logger.info(
          "AutoGluon trained: %d models, best=%s (MAE=%.1f)",
          n_models, best_model, best_score,
      )
      
      return predictor
  ```
- **PATTERN**: Mirror `train_xgboost()` signature (Polars in, model out)
- **GOTCHA**: Lazy import `from autogluon.tabular import TabularPredictor` inside function to avoid import overhead when not using AutoGluon
- **GOTCHA**: `mlflow.autolog(disable=True)` MUST be called before fit, not before import
- **GOTCHA**: Score sign: negate `score_val` when it's MAE (higher-is-better convention)
- **VALIDATE**: `uv run ruff check src/windcast/models/autogluon_model.py`

### Task 3: CREATE `scripts/train_autogluon.py`

- **IMPLEMENT**: Training script mirroring `scripts/train.py` structure
- **KEY DIFFERENCES from train.py**:
  1. No `mlflow.xgboost.autolog()` — instead `mlflow.autolog(disable=True)` before fit
  2. After each horizon fit, log: MAE, RMSE, skill_score (from `compute_metrics`), plus AutoGluon leaderboard as artifact
  3. In parent run: log leaderboard summary, feature importance, total fit time
  4. Tag: `enercast.backend = "autogluon"`
  5. Use same `_temporal_split`, `_resolve_horizon_features`, `_build_horizon_target` logic — import from a shared location OR copy (prefer copy to keep scripts self-contained, matching existing pattern)
- **IMPORTS**: Same as train.py plus `from windcast.models.autogluon_model import AutoGluonConfig, train_autogluon`
- **MLflow structure**:
  ```
  Parent run: "{run_label}-autogluon-{feature_set}"
    ├── Child run: "h01" (horizon 1)
    │   ├── metrics: h1_mae, h1_rmse, h1_skill_score, h1_bias
    │   ├── params: horizon_steps, n_features, n_ag_models, best_ag_model
    │   └── artifact: ag_leaderboard_h1.csv
    ├── Child run: "h06" (horizon 6)
    │   └── ...
    └── ...
  Parent metrics (bubbled up): h1_mae, h1_skill_score, h6_mae, ...
  Parent artifact: ag_leaderboard_summary.csv
  Parent description: Markdown with results table + AutoGluon model zoo summary
  ```
- **VALIDATE**: `uv run ruff check scripts/train_autogluon.py && uv run pyright scripts/train_autogluon.py`

### Task 4: CREATE `tests/models/test_autogluon_model.py`

- **IMPLEMENT**:
  - `TestAutoGluonConfig`: test defaults, test custom values
  - `TestTrainAutoGluon`: test with small synthetic data (reuse `_make_regression_data` pattern from test_xgboost_model.py). Use `time_limit=30` and `presets="medium_quality"` for fast tests.
  - Mark test with `@pytest.mark.slow` if it takes >10s (AutoGluon fit is inherently slower)
- **PATTERN**: Mirror `tests/models/test_xgboost_model.py`
- **GOTCHA**: AutoGluon fit creates temp directories — use `tmp_path` fixture for cleanup
- **VALIDATE**: `uv run pytest tests/models/test_autogluon_model.py -v`

### Task 5: UPDATE `src/windcast/models/__init__.py`

- **IMPLEMENT**: Add AutoGluon imports and exports
  ```python
  from windcast.models.autogluon_model import AutoGluonConfig, train_autogluon
  ```
  Add to `__all__`.
- **VALIDATE**: `uv run python -c "from windcast.models import AutoGluonConfig, train_autogluon"`

### Task 6: Full validation

- **VALIDATE**: Run full validation suite
  ```bash
  uv run ruff check src/ tests/ scripts/
  uv run ruff format --check src/ tests/ scripts/
  uv run pyright src/
  uv run pytest tests/ -v
  ```

---

## TESTING STRATEGY

### Unit Tests

- `AutoGluonConfig`: defaults, custom values, serialization
- `train_autogluon`: returns TabularPredictor, predictions have correct shape, MLflow autolog is restored after fit

### Integration Tests (manual — during training run)

- `scripts/train_autogluon.py --turbine-id kwf1 --feature-set wind_full --horizons 1 6 12 24 48`
- Verify MLflow UI shows: parent run + 5 child runs, leaderboard artifacts, comparable metrics

### Edge Cases

- Empty validation set → should error gracefully
- Missing feature columns → should warn and continue (same as train.py)
- AutoGluon fit timeout → should still return best model so far

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
uv run pytest tests/models/test_autogluon_model.py -v
uv run pytest tests/ -v  # full suite — no regressions
```

### Level 3: Integration Test (manual)

```bash
# Train AutoGluon on wind_full features (same as XGBoost)
uv run python scripts/train_autogluon.py \
    --turbine-id kwf1 \
    --feature-set wind_full \
    --horizons 1 6 12 24 48

# Compare in MLflow UI
mlflow ui
# → Look for autogluon run alongside xgboost runs in enercast-kelmarsh experiment
```

### Level 4: Comparison Validation

In MLflow UI, verify:
- AutoGluon parent run has h{n}_mae and h{n}_skill_score metrics
- Leaderboard CSV artifacts present in each child run
- Results are in the same ballpark as XGBoost (±20%)
- Description shows model zoo summary

---

## ACCEPTANCE CRITERIA

- [ ] `autogluon.tabular[all]` added to pyproject.toml, `uv sync` works
- [ ] `autogluon_model.py` created with AutoGluonConfig + train_autogluon
- [ ] `train_autogluon.py` runs end-to-end with same CLI interface as train.py
- [ ] MLflow shows AutoGluon runs with per-horizon metrics, leaderboard artifacts
- [ ] Comparison possible: XGBoost vs AutoGluon on same feature set in MLflow UI
- [ ] All tests pass (existing + new), ruff + pyright clean
- [ ] Can state: "adding a new ML backend = 1 model file + 1 script, zero pipeline changes"

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] `uv sync` successful with autogluon dependency
- [ ] `uv run ruff check src/ tests/ scripts/` — 0 errors
- [ ] `uv run ruff format --check src/ tests/ scripts/` — 0 errors
- [ ] `uv run pyright src/` — 0 errors
- [ ] `uv run pytest tests/ -v` — all pass
- [ ] Integration run: `train_autogluon.py` completes, MLflow populated
- [ ] All acceptance criteria met

---

## NOTES

### Design Decisions

1. **AutoGluon-Tabular, not TimeSeries**: Tabular uses the same pre-computed feature Parquets as XGBoost. TimeSeries requires TimeSeriesDataFrame format + handles lags internally (different paradigm, more integration work, less transparent).

2. **No custom MLflow flavor**: MLflow declined to add an AutoGluon flavor (issue #13214). Creating a community flavor is possible but overkill for the demo. Manual logging gives us full control and the same MLflow UI experience.

3. **Lazy import**: `from autogluon.tabular import TabularPredictor` inside the function, not at module level. AutoGluon is heavy (~500MB) and we don't want to slow down imports for users not using it.

4. **Script copy pattern**: `train_autogluon.py` copies `_temporal_split` and `_build_horizon_target` from `train.py` rather than extracting to a shared module. This matches the existing pattern (train_mlforecast.py also copies `_temporal_split`). Refactoring to shared utils is a post-demo improvement.

5. **`best_quality` preset**: Triggers bagging + stacking (WeightedEnsemble_L2). More impressive for demo than `medium_quality`. `time_limit=300` (5 min) per horizon keeps total time reasonable (~25 min for 5 horizons).

6. **Excluded GPU models**: `NN_TORCH`, `FASTAI`, `TABPFNV2`, `TabICL`, `MITRA` excluded — no GPU on macOS. The remaining models (LightGBM, XGBoost, CatBoost, RF, ExtraTrees, KNN, LR) are sufficient and match WN's own stack.

### Risks

- **Install size**: `autogluon.tabular[all]` is ~500MB. `uv sync` will take a few minutes.
- **Training time**: `best_quality` with bagging is slow. 5 min × 5 horizons = ~25 min total. Acceptable for a one-time demo run.
- **macOS LightGBM**: May segfault with wrong libomp. If this happens, install `brew install libomp` or fallback to `presets="medium_quality"`.

### Confidence Score: 8/10

High confidence because:
- Pattern is well-established (3rd script following same template)
- AutoGluon API is simple (5 lines of core code)
- No architectural changes needed
- Main risk is install/compatibility issues on macOS
