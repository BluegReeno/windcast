# Feature: Training Harness + MLflow Lineage Tags

The following plan should be complete, but validate documentation and codebase patterns before implementing.

## Feature Description

Refactor the duplicated training scripts (`train.py` + `train_autogluon.py`) into a single unified training harness with a Backend Protocol, and add MLflow lineage tags (generation, data_quality, nwp_source, git state) that are logged automatically for all future runs. Then backfill existing runs with the same tags.

## User Story

As an ML engineer iterating on EnerCast models,
I want a single training entry point that handles all MLflow plumbing regardless of backend,
So that adding a new ML backend = implementing 6 methods, and every run is automatically tagged with lineage metadata for audit and filtering.

## Problem Statement

1. **Code duplication**: `train.py` (515 lines) and `train_autogluon.py` (530 lines) share ~90% identical code — `DOMAIN_CONFIG`, `_temporal_split()`, `_resolve_horizon_features()`, `_build_horizon_target()`, CLI args, MLflow setup, parent/child run structure, metric logging, descriptions. Only ~50 lines per script are truly backend-specific.

2. **Missing lineage metadata**: After 3 generations of runs (Gen1 ERA5+AG leak, Gen2 ERA5 leak, Gen3 clean), runs are indistinguishable in MLflow. No tags for generation, data quality, NWP source, or git state.

3. **Barrier to new backends**: Adding a backend means copy-pasting 500+ lines and editing ~50. Should be: implement a protocol, pass `--backend name`.

## Solution Statement

- **Backend Protocol**: 6-method interface (`train`, `predict`, `mlflow_setup`, `extra_params`, `log_child_artifacts`, `describe_model`). XGBoost and AutoGluon implement it.
- **Training Harness**: Single `run_training()` function that owns the entire loop (data load → split → MLflow parent → horizon loop → child runs → metrics → descriptions). Backends plug in.
- **Lineage Tags**: Auto-logged on every run — git commit/branch/dirty, `enercast.generation`, `enercast.data_quality`, `enercast.nwp_source`. Default: `CLEAN` + auto-detected NWP source.
- **Unified CLI**: `scripts/train.py --backend xgboost|autogluon [--generation gen4] [--nwp-source forecast]`
- **Backfill script**: One-shot annotation of existing runs by date classification.
- **compare_runs.py filter**: Default to `data_quality = CLEAN`, `--all` to include everything.

## Feature Metadata

**Feature Type**: Refactor + Enhancement
**Estimated Complexity**: Medium
**Primary Systems Affected**: `scripts/train.py`, `scripts/train_autogluon.py`, `src/windcast/tracking/`, `scripts/compare_runs.py`
**Dependencies**: No new dependencies (gitpython already available via mlflow transitive)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — MUST READ BEFORE IMPLEMENTING

- `scripts/train.py` (full file) — XGBoost training loop, the "source of truth" for the harness structure
- `scripts/train_autogluon.py` (full file) — AutoGluon training loop, near-identical structure, shows backend-specific delta
- `scripts/log_tso_baseline.py` (full file) — TSO benchmark script, stays separate (no training), but should get lineage tags too
- `scripts/compare_runs.py` (lines 96-108, 302-327) — `_fetch_parent_runs()` filter string + `_print_markdown_table()`, need `data_quality` filter
- `src/windcast/tracking/mlflow_utils.py` (full file) — `setup_mlflow()`, `log_evaluation_results()`, `log_stepped_horizon_metrics()`, `STEPPED_METRIC_MAP`
- `src/windcast/tracking/__init__.py` — current exports
- `src/windcast/models/xgboost_model.py` — `XGBoostConfig`, `train_xgboost()` signature: `(X_train: pl.DataFrame, y_train: pl.Series, X_val, y_val, config) -> xgb.XGBRegressor`
- `src/windcast/models/autogluon_model.py` — `AutoGluonConfig`, `train_autogluon()` signature: `(X_train, y_train, X_val, y_val, config, ag_path) -> TabularPredictor`. Note: calls `mlflow.autolog(disable=True)` internally
- `src/windcast/models/evaluation.py` — `compute_metrics()`, `compute_persistence_metrics()`
- `src/windcast/features/__init__.py` — `get_feature_set()`, `list_feature_sets()`
- `src/windcast/config.py` — `get_settings()`, `EnerCastSettings` with `train_years`, `val_years`, `forecast_horizons`, `mlflow_tracking_uri`, `features_dir`
- `tests/models/test_autogluon_model.py` (lines 107-112) — leak guard test pattern

### New Files to Create

- `src/windcast/training/__init__.py` — exports `TrainingBackend`, `run_training`
- `src/windcast/training/harness.py` — `TrainingBackend` Protocol + `run_training()` function + shared utilities (`_temporal_split`, `_resolve_horizon_features`, `_build_horizon_target`, `_build_horizon_desc`)
- `src/windcast/training/backends.py` — `XGBoostBackend` + `AutoGluonBackend` implementations
- `src/windcast/training/lineage.py` — `log_lineage_tags()` helper (git state + enercast.* lineage tags)
- `scripts/annotate_mlflow_runs.py` — one-shot backfill script for existing runs
- `tests/training/__init__.py`
- `tests/training/test_harness.py` — test shared utilities + harness with mock backend
- `tests/training/test_lineage.py` — test git tag capture + lineage tag generation

### Files to Modify

- `scripts/train.py` — gut and replace with thin CLI that instantiates backend + calls `run_training()`
- `scripts/log_tso_baseline.py` — add lineage tags (same `log_lineage_tags()` call)
- `scripts/compare_runs.py` — add `--data-quality` filter, `--all` flag
- `src/windcast/tracking/__init__.py` — re-export new functions if needed

### Files to Delete

- `scripts/train_autogluon.py` — replaced by `scripts/train.py --backend autogluon`

### Patterns to Follow

**Naming:** snake_case functions, PascalCase classes, `UPPER_SNAKE` constants
**Imports:** stdlib → third-party → local, sorted alphabetically
**Error handling:** fail fast with logger.error + return, no silent fallbacks
**MLflow tags:** `enercast.*` namespace (existing convention)
**Protocol pattern:** `typing.Protocol` with `runtime_checkable=False` (perf)

**Existing tag taxonomy (preserved):**
```python
{
    "enercast.stage": "dev",
    "enercast.domain": domain,
    "enercast.purpose": "baseline",
    "enercast.backend": backend_name,
    "enercast.data_resolution_min": str(resolution),
    "enercast.feature_set": feature_set,
    "enercast.run_type": "parent" | "child",
    # Child-only:
    "enercast.horizon_steps": str(h),
    "enercast.horizon_desc": description,
}
```

**New lineage tags (added by this feature):**
```python
{
    "enercast.generation": "gen3",          # CLI arg, semi-manual
    "enercast.nwp_source": "forecast",      # auto-detected from features or CLI
    "enercast.data_quality": "CLEAN",       # default, override via CLI
    "enercast.change_reason": "...",        # optional CLI arg, free text
    "mlflow.source.git.branch": "main",     # auto from git
    "enercast.git.dirty": "false",          # auto from git
}
```
Note: `mlflow.source.git.commit` is auto-set by MLflow when running a .py script from a git repo. We only need to add branch and dirty.

---

## IMPLEMENTATION PLAN

### Phase 1: Training Package Foundation

Create `src/windcast/training/` with the Backend Protocol, shared utilities, and lineage helper. No script changes yet — this is pure library code.

### Phase 2: Backend Implementations

Implement `XGBoostBackend` and `AutoGluonBackend` as thin wrappers around existing model functions. Verify they match the current behavior exactly.

### Phase 3: Training Harness

Port the shared training loop from `train.py` into `run_training()`. This is the biggest task — it absorbs ~450 lines of currently-duplicated logic.

### Phase 4: Unified CLI

Rewrite `scripts/train.py` as a thin CLI (~80 lines) that parses args, picks a backend, calls `run_training()`. Delete `scripts/train_autogluon.py`.

### Phase 5: Lineage Integration

Wire `log_lineage_tags()` into the harness and into `log_tso_baseline.py`. Add CLI args for `--generation`, `--nwp-source`, `--data-quality`, `--change-reason`.

### Phase 6: compare_runs.py Filter

Add `--data-quality` / `--all` flags to `compare_runs.py`.

### Phase 7: Backfill Script

Write `scripts/annotate_mlflow_runs.py` to tag existing runs by date.

### Phase 8: Tests + Validation

Unit tests for utilities, lineage, and a mock-backend harness test. Run full validation suite.

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/windcast/training/__init__.py`

- **IMPLEMENT**: Export `TrainingBackend`, `run_training` (will be created in next tasks)
- **VALIDATE**: `uv run python -c "from windcast.training import TrainingBackend, run_training"` (will fail until Task 3, that's OK)

### Task 2: CREATE `src/windcast/training/harness.py` — Protocol + Utilities

- **IMPLEMENT**: 
  1. `TrainingBackend` Protocol:
     ```python
     from typing import Any, Protocol
     import numpy as np
     import polars as pl
     
     class TrainingBackend(Protocol):
         @property
         def name(self) -> str: ...
         
         def mlflow_setup(self) -> None:
             """Backend-specific MLflow config (e.g., XGBoost autolog)."""
             ...
         
         def extra_params(self) -> dict[str, Any]:
             """Backend-specific MLflow params (e.g., AG presets, time_limit)."""
             ...
         
         def train(
             self,
             X_train: pl.DataFrame,
             y_train: pl.Series,
             X_val: pl.DataFrame,
             y_val: pl.Series,
         ) -> Any:
             """Train model and return fitted object."""
             ...
         
         def predict(self, model: Any, X: pl.DataFrame) -> np.ndarray:
             """Generate predictions from fitted model."""
             ...
         
         def log_child_artifacts(self, model: Any, horizon: int) -> None:
             """Log backend-specific artifacts on the child run (e.g., AG leaderboard)."""
             ...
         
         def describe_model(self, model: Any) -> str:
             """One-line model description for Markdown notes (e.g., 'Trees: 142' or 'Best: WeightedEnsemble')."""
             ...
     ```
  2. Move from `train.py` (identical in both scripts):
     - `DOMAIN_CONFIG` dict
     - `temporal_split()` (rename: drop leading `_`, it's now a public API)
     - `resolve_horizon_features()` (same rename)
     - `build_horizon_target()` (same rename)
     - `build_horizon_desc()` — extract the horizon_desc computation into a helper
  3. Do NOT implement `run_training()` yet (Task 5)
- **PATTERN**: Mirror function signatures from `scripts/train.py` lines 36-117
- **GOTCHA**: `_build_horizon_target` has default `target_col_name="active_power_kw"` — make it required param in the public version (it's always passed explicitly in both scripts)
- **VALIDATE**: `uv run python -c "from windcast.training.harness import TrainingBackend, DOMAIN_CONFIG, temporal_split"`

### Task 3: CREATE `src/windcast/training/lineage.py`

- **IMPLEMENT**:
  ```python
  def get_git_info() -> dict[str, str]:
      """Return git commit, branch, dirty state as a dict of MLflow tags."""
  ```
  Use `subprocess.run(["git", ...])` for commit, branch, dirty — avoid importing gitpython (subprocess is lighter and already available). Three calls:
  - `git rev-parse HEAD` → commit hash
  - `git rev-parse --abbrev-ref HEAD` → branch
  - `git diff --quiet` → exit code 0 = clean, 1 = dirty
  
  ```python
  def log_lineage_tags(
      generation: str | None = None,
      nwp_source: str = "forecast",
      data_quality: str = "CLEAN",
      change_reason: str | None = None,
  ) -> None:
      """Log lineage + git tags on the active MLflow run."""
  ```
  Sets:
  - `enercast.generation` (if provided)
  - `enercast.nwp_source`
  - `enercast.data_quality`
  - `enercast.change_reason` (if provided)
  - `mlflow.source.git.branch`
  - `enercast.git.dirty`
  
  Note: `mlflow.source.git.commit` is auto-set by MLflow — don't duplicate it.
- **VALIDATE**: `uv run python -c "from windcast.training.lineage import get_git_info; print(get_git_info())"`

### Task 4: CREATE `src/windcast/training/backends.py`

- **IMPLEMENT**:
  1. `XGBoostBackend`:
     ```python
     class XGBoostBackend:
         def __init__(self, config: XGBoostConfig | None = None):
             self._config = config or XGBoostConfig()
         
         @property
         def name(self) -> str:
             return "xgboost"
         
         def mlflow_setup(self) -> None:
             import mlflow.xgboost
             mlflow.xgboost.autolog(
                 log_datasets=False, log_models=False, log_model_signatures=False
             )
         
         def extra_params(self) -> dict[str, Any]:
             return {}  # XGBoost params are auto-logged via autolog
         
         def train(self, X_train, y_train, X_val, y_val) -> Any:
             return train_xgboost(X_train, y_train, X_val, y_val, self._config)
         
         def predict(self, model, X) -> np.ndarray:
             return model.predict(X)
         
         def log_child_artifacts(self, model, horizon) -> None:
             pass  # autolog handles everything
         
         def describe_model(self, model) -> str:
             best_iter = getattr(model, "best_iteration", "?")
             return f"Trees: {best_iter}"
     ```
  2. `AutoGluonBackend`:
     ```python
     class AutoGluonBackend:
         def __init__(self, config: AutoGluonConfig | None = None, ag_base_path: Path | None = None):
             self._config = config or AutoGluonConfig()
             self._ag_base_path = ag_base_path
             self._current_path: Path | None = None  # set per-horizon in train()
         
         @property
         def name(self) -> str:
             return "autogluon"
         
         def mlflow_setup(self) -> None:
             pass  # autogluon_model.py disables autolog internally
         
         def extra_params(self) -> dict[str, Any]:
             return {
                 "ag.presets": self._config.presets,
                 "ag.time_limit": self._config.time_limit,
                 "ag.eval_metric": self._config.eval_metric,
             }
         
         def train(self, X_train, y_train, X_val, y_val) -> Any:
             import tempfile
             self._current_path = Path(tempfile.mkdtemp(prefix="ag_"))
             return train_autogluon(
                 X_train, y_train, X_val, y_val, self._config, ag_path=self._current_path
             )
         
         def predict(self, model, X) -> np.ndarray:
             return model.predict(X.to_pandas()).values
         
         def log_child_artifacts(self, model, horizon) -> None:
             import mlflow
             lb = model.leaderboard(data=None, silent=True)
             if self._current_path:
                 lb_path = self._current_path / f"ag_leaderboard_h{horizon}.csv"
                 lb.to_csv(str(lb_path), index=False)
                 mlflow.log_artifact(str(lb_path))
                 best_model = lb.iloc[0]["model"]
                 n_models = len(lb)
                 mlflow.log_params({"n_ag_models": n_models, "best_ag_model": best_model})
         
         def describe_model(self, model) -> str:
             lb = model.leaderboard(data=None, silent=True)
             return f"Best: {lb.iloc[0]['model']}, {len(lb)} models"
     ```
- **IMPORTS**: `from windcast.models.xgboost_model import XGBoostConfig, train_xgboost` + `from windcast.models.autogluon_model import AutoGluonConfig, train_autogluon`
- **GOTCHA**: AutoGluon `predict()` requires pandas conversion — this is already handled in `predict()` method, NOT in the harness
- **GOTCHA**: AG leaderboard calls `model.leaderboard(data=None)` — don't pass validation data (that was the tuning leak)
- **VALIDATE**: `uv run python -c "from windcast.training.backends import XGBoostBackend, AutoGluonBackend; print('OK')"`

### Task 5: IMPLEMENT `run_training()` in `src/windcast/training/harness.py`

- **IMPLEMENT**: The main training harness function. Port from `scripts/train.py` lines 120-511, but parametrized over `TrainingBackend`:
  ```python
  def run_training(
      backend: TrainingBackend,
      domain: str,
      dataset: str,
      feature_set_name: str,
      features_path: Path,
      experiment_name: str,
      horizons: list[int],
      turbine_id: str = "kwf1",
      # Lineage args
      generation: str | None = None,
      nwp_source: str = "forecast",
      data_quality: str = "CLEAN",
      change_reason: str | None = None,
  ) -> None:
  ```
  Key structure:
  1. Load features parquet, resolve feature columns
  2. `temporal_split()` 
  3. `backend.mlflow_setup()`
  4. Parent run: tags (existing + lineage via `log_lineage_tags()`), params (shared + `backend.extra_params()`), dataset provenance (`mlflow.log_input`)
  5. Horizon loop:
     - `resolve_horizon_features()` + `build_horizon_target()`
     - Child run with tags
     - `backend.train()` → `backend.predict()` → `compute_metrics()` → `log_evaluation_results()`
     - Test metrics (same pattern)
     - `backend.log_child_artifacts()`
     - Child description with `backend.describe_model()`
  6. `log_stepped_horizon_metrics()` on parent
  7. Bubble up child metrics to parent
  8. Parent description
  
  **Run name**: `{run_label}-{feature_set}` for XGBoost, `{run_label}-autogluon-{feature_set}` for AG → generalize to `{run_label}-{backend.name}-{feature_set}` (but keep `{run_label}-{feature_set}` when backend is "xgboost" for backward compat with compare_runs label parsing? Actually no — just use consistent naming. `compare_runs.py` reads `tags.enercast.backend` not the run name).
  
  Actually, use: `{run_label}-{feature_set}` when backend is xgboost (historical convention), `{run_label}-{backend.name}-{feature_set}` otherwise.
  Wait, let's simplify: always `{run_label}-{backend.name}-{feature_set}`. This is cleaner. compare_runs uses tags, not run names.

- **PATTERN**: Follow the exact metric computation and logging pattern from `train.py` lines 349-477 (val metrics → test metrics → horizon_metrics stash → child description)
- **GOTCHA**: `data_resolution` = 10 for wind, 60 for demand/solar — derive from domain, don't add a param
- **GOTCHA**: feature file path resolution differs by domain: wind uses `kelmarsh_{turbine_id}.parquet`, demand/solar use `{dataset}_features.parquet`. Keep this logic in the harness.
- **VALIDATE**: Unit tests (Task 9) — actual training test would require data, so defer to integration

### Task 6: REWRITE `scripts/train.py` as thin CLI

- **IMPLEMENT**: ~80 lines total. Structure:
  ```python
  def main():
      parser = argparse.ArgumentParser(...)
      parser.add_argument("--backend", choices=["xgboost", "autogluon"], default="xgboost")
      # Shared args: --domain, --features-dir, --feature-set, --dataset, --turbine-id,
      #              --experiment-name, --horizons
      # AutoGluon-specific: --time-limit, --presets
      # Lineage args: --generation, --nwp-source, --data-quality, --change-reason
      args = parser.parse_args()
      
      # Build backend
      if args.backend == "xgboost":
          backend = XGBoostBackend()
      elif args.backend == "autogluon":
          backend = AutoGluonBackend(
              config=AutoGluonConfig(presets=args.presets, time_limit=args.time_limit)
          )
      
      # Call harness
      run_training(
          backend=backend,
          domain=args.domain,
          dataset=dataset,
          feature_set_name=feature_set,
          features_path=parquet_path,
          experiment_name=experiment_name,
          horizons=horizons,
          turbine_id=args.turbine_id,
          generation=args.generation,
          nwp_source=args.nwp_source,
          data_quality=args.data_quality,
          change_reason=args.change_reason,
      )
  ```
  Keep the domain-specific defaults (feature set, dataset) in the CLI, not in the harness.
- **PATTERN**: Mirror current CLI arg names exactly for backward compat
- **GOTCHA**: `--presets` and `--time-limit` only relevant for autogluon — don't error if passed to xgboost (just ignore)
- **VALIDATE**: `uv run python scripts/train.py --help` shows all args including `--backend` and lineage args

### Task 7: DELETE `scripts/train_autogluon.py`

- **VALIDATE**: `uv run python scripts/train.py --backend autogluon --help` works
- **VALIDATE**: Grep codebase for references to `train_autogluon.py` and update any docs/comments

### Task 8: UPDATE `scripts/log_tso_baseline.py` — add lineage tags

- **IMPLEMENT**: Import and call `log_lineage_tags()` inside the `mlflow.start_run()` block. Add CLI args for `--generation`, `--data-quality`.
- **PATTERN**: Same tag set as harness (except `nwp_source` = N/A for TSO)
- **VALIDATE**: `uv run python scripts/log_tso_baseline.py --help` shows lineage args

### Task 9: UPDATE `scripts/compare_runs.py` — data_quality filter

- **IMPLEMENT**:
  1. Add `--data-quality` arg (default: `"CLEAN"`) and `--all` flag
  2. Update `_fetch_parent_runs()` filter string:
     ```python
     filter_str = "tags.`enercast.run_type` = 'parent'"
     if not args.all and args.data_quality:
         filter_str += f" AND tags.`enercast.data_quality` = '{args.data_quality}'"
     ```
  3. Handle backward compat: runs without `data_quality` tag won't match the filter. Solution: `--all` is needed to see old runs, OR the backfill script (Task 10) tags them first.
- **VALIDATE**: `uv run python scripts/compare_runs.py --help` shows `--data-quality` and `--all`

### Task 10: CREATE `scripts/annotate_mlflow_runs.py` — backfill

- **IMPLEMENT**: One-shot script:
  ```python
  """Annotate existing MLflow runs with lineage tags (generation, data_quality, nwp_source).
  
  Run once after implementing the new tag taxonomy. Classifies runs by creation
  date into generations and marks leaked runs.
  """
  ```
  Logic:
  1. Connect to MLflow, query all parent runs from `enercast-rte_france` and `enercast-kelmarsh`
  2. For each run, read `start_time` and classify:
     - Before 2026-04-09 23:00 UTC → Gen 1 (ERA5 + AG tuning leak)
     - 2026-04-09 23:00 – 2026-04-10 06:00 → Gen 2 (ERA5 leak only)
     - After 2026-04-10 06:00 → Gen 3 (clean)
  3. Set tags:
     - Gen 1: `generation=gen1`, `data_quality=LEAKED`, `nwp_source=era5`, `change_reason=initial_run_era5_and_ag_leak`
     - Gen 2: `generation=gen2`, `data_quality=LEAKED`, `nwp_source=era5`, `change_reason=fixed_ag_tuning_leak_era5_still_present`
     - Gen 3: `generation=gen3`, `data_quality=CLEAN`, `nwp_source=forecast`, `change_reason=ported_wattcast_forecast_provider`
  4. Add `mlflow.note.content` Markdown annotation explaining the classification
  5. Rename run: prepend `[LEAKED] ` to run name for Gen 1/2
  6. Also tag child runs of each parent with the same lineage tags
  7. Print summary table to stdout
  
  The date boundaries should be CLI args with defaults (for reusability), but the defaults match our actual timeline.
- **GOTCHA**: `client.set_tag()` works on finished runs — confirmed by research
- **GOTCHA**: Rename via `client.set_tag(run_id, "mlflow.runName", f"[LEAKED] {old_name}")` — this is the correct way
- **VALIDATE**: `uv run python scripts/annotate_mlflow_runs.py --dry-run` shows what would be changed without writing

### Task 11: UPDATE `src/windcast/training/__init__.py`

- **IMPLEMENT**: Final exports:
  ```python
  from windcast.training.backends import AutoGluonBackend, XGBoostBackend
  from windcast.training.harness import TrainingBackend, run_training
  from windcast.training.lineage import get_git_info, log_lineage_tags
  ```
- **VALIDATE**: `uv run python -c "from windcast.training import TrainingBackend, run_training, XGBoostBackend, AutoGluonBackend, log_lineage_tags"`

### Task 12: CREATE `tests/training/__init__.py` + `tests/training/test_harness.py`

- **IMPLEMENT**:
  1. Test `temporal_split()`: verify train/val/test boundaries on synthetic data
  2. Test `resolve_horizon_features()`: NWP column resolution, fallback behavior
  3. Test `build_horizon_target()`: target shift, null removal, rename_map
  4. Test `build_horizon_desc()`: minute/hour/day formatting
  5. Test `run_training()` with a mock backend: create a `MockBackend` that returns dummy predictions. Verify MLflow tags are set correctly (lineage tags present, parent/child structure). Use `mlflow.set_tracking_uri("sqlite:///:memory:")` or temp dir for isolation.
- **PATTERN**: Follow `tests/models/test_autogluon_model.py` for MLflow test patterns
- **VALIDATE**: `uv run pytest tests/training/test_harness.py -v`

### Task 13: CREATE `tests/training/test_lineage.py`

- **IMPLEMENT**:
  1. Test `get_git_info()`: returns dict with commit, branch, dirty keys (run from inside git repo)
  2. Test `log_lineage_tags()`: verify tags set on active MLflow run. Use temp tracking URI.
- **VALIDATE**: `uv run pytest tests/training/test_lineage.py -v`

### Task 14: UPDATE docs and STATUS.md references

- **IMPLEMENT**:
  1. Grep for `train_autogluon.py` references in docs — update to `train.py --backend autogluon`
  2. Update `README.md` quick start if it references `train_autogluon.py`
  3. Update `.claude/STATUS.md` to note refactoring is done
  4. Update `docs/mlflow-ui-setup.md` with new filter recipes (`tags.enercast.data_quality = "CLEAN"`)
- **VALIDATE**: `grep -r "train_autogluon" docs/ .claude/ README.md` returns zero matches

---

## TESTING STRATEGY

### Unit Tests

- `tests/training/test_harness.py`: shared utilities (temporal_split, resolve_horizon_features, build_horizon_target) + mock-backend integration
- `tests/training/test_lineage.py`: git info capture + MLflow tag logging

### Integration Test

- Run `uv run python scripts/train.py --backend xgboost --domain wind --feature-set wind_baseline --horizons 1` on actual data (if available) and verify MLflow run has all expected tags
- This is manual / CI — not a pytest test

### Edge Cases

- Backend with no extra_params (XGBoost returns `{}`)
- Backend with no child artifacts (XGBoost `log_child_artifacts` is a no-op)
- Missing NWP features (feature resolution fallback)
- Empty test set (some datasets don't have enough data for 3-way split)
- Git state when not in a repo (should not crash, just skip git tags)
- `--all` flag in compare_runs shows runs both with and without `data_quality` tag

---

## VALIDATION COMMANDS

### Level 1: Lint + Type Check

```bash
uv run ruff check src/windcast/training/ tests/training/ scripts/train.py scripts/compare_runs.py scripts/log_tso_baseline.py
uv run ruff format --check src/windcast/training/ tests/training/ scripts/train.py scripts/compare_runs.py scripts/log_tso_baseline.py
uv run pyright src/windcast/training/
```

### Level 2: Unit Tests

```bash
uv run pytest tests/training/ -v
uv run pytest tests/ -v  # full suite — verify no regressions
```

### Level 3: CLI Smoke Tests

```bash
uv run python scripts/train.py --help
uv run python scripts/train.py --backend autogluon --help
uv run python scripts/compare_runs.py --help
uv run python scripts/annotate_mlflow_runs.py --dry-run
```

### Level 4: Integration (manual, requires data)

```bash
# XGBoost training with lineage tags
uv run python scripts/train.py --backend xgboost --domain demand --dataset rte_france --feature-set demand_full --generation gen4 --nwp-source forecast

# AutoGluon training (same CLI!)
uv run python scripts/train.py --backend autogluon --domain demand --dataset rte_france --feature-set demand_full --generation gen4 --presets good_quality --time-limit 60

# compare with filter
uv run python scripts/compare_runs.py --experiment enercast-rte_france --data-quality CLEAN

# backfill
uv run python scripts/annotate_mlflow_runs.py
```

---

## ACCEPTANCE CRITERIA

- [ ] `scripts/train_autogluon.py` deleted, `train.py --backend autogluon` produces identical MLflow output
- [ ] All runs automatically get lineage tags (generation, data_quality, nwp_source, git state)
- [ ] `compare_runs.py` defaults to showing only CLEAN runs, `--all` shows everything
- [ ] Existing runs backfilled with generation + data_quality tags
- [ ] Adding a hypothetical new backend requires only: 1 class in `backends.py` implementing 6 methods + 1 `elif` in `train.py` CLI
- [ ] All 267+ existing tests still pass
- [ ] New tests cover utilities, lineage, and mock-backend harness
- [ ] `ruff check` + `ruff format --check` + `pyright` clean
- [ ] No behavior change in MLflow output (same metric names, same tag names, same parent/child structure)

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order (1-14)
- [ ] Validation commands pass:
  - [ ] ruff check + format clean
  - [ ] pyright clean
  - [ ] pytest full suite passes
  - [ ] CLI smoke tests work
- [ ] `train_autogluon.py` deleted, no dangling references
- [ ] Backfill script tested with `--dry-run`
- [ ] All acceptance criteria met

---

## NOTES

### Design Decisions

1. **Protocol, not ABC**: `TrainingBackend` is a `typing.Protocol` — no inheritance required. Backends are standalone classes. This is more Pythonic and doesn't force import of heavy libs (AutoGluon) when only using XGBoost.

2. **Harness owns MLflow, backends don't**: Backends never call `mlflow.start_run()` or set tags. They only log backend-specific artifacts/params when asked. This ensures consistent tagging across all backends.

3. **TSO baseline stays separate**: `log_tso_baseline.py` has no training loop — it just evaluates an existing column. Forcing it into the harness would add complexity for no gain. It gets lineage tags via the shared `log_lineage_tags()` helper.

4. **Semi-manual generation**: `--generation` is a CLI arg, not auto-detected. Reason: generation boundaries are semantic (what changed), not mechanical (git commit). Auto-incrementing from MLflow would give meaningless numbers. The engineer decides "this is gen4 because I changed X".

5. **`nwp_source` auto-detection**: If features contain `nwp_*` columns and no explicit `--nwp-source` is given, default to `"forecast"`. If no NWP columns → `"none"`. This covers 90% of cases without manual input.

6. **Run naming convention**: `{label}-{backend}-{feature_set}` for all backends (not just AG). This breaks backward compat with old run names (`kwf1-wind_full` → `kwf1-xgboost-wind_full`) but compare_runs.py uses tags, not names, so no impact on tooling. Clearer in the UI.

### Risks

- **AutoGluon import time**: Importing `autogluon.tabular` takes ~5 seconds. The backend is only imported when `--backend autogluon` is used, so XGBoost runs are not slowed. Ensure the import is inside the `AutoGluonBackend` class or guarded by a conditional.
- **Metric parity**: The refactored harness must produce bit-identical metrics and tags vs the current scripts. Test by running both old and new on the same data and diffing MLflow output.
