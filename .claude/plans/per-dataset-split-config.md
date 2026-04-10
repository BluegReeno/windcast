# Feature: Per-Dataset Split Configuration + CLI Override

## Feature Description

Move `train_years`/`val_years` from global `WindCastSettings` to per-dataset configs, add CLI overrides to all training scripts, and consolidate duplicated `temporal_split()` logic. This eliminates the need for `WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2` env-var prefixes when running demand pipelines.

## User Story

As an ML engineer running training pipelines across multiple domains,
I want each dataset to carry its natural temporal split configuration,
So that `uv run python scripts/train.py --domain demand --dataset rte_france` just works with the right split, and I can override with `--train-years 6` to experiment.

## Problem Statement

- `train_years`/`val_years` are global settings (default 5/1), but optimal splits are per-dataset (Kelmarsh 8y data → 5/1/2, RTE France 11y data → 8/2/1)
- No CLI args for split config — must use env vars `WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2`
- Not reproducible (split depends on shell env, not the command)
- `temporal_split()` is duplicated in 4 places with minor column-name variations
- `train_years`/`val_years` counts are NOT logged to MLflow (only computed boundaries are)

## Solution Statement

1. Add `train_years`/`val_years` fields to all 3 dataset config classes with per-dataset defaults
2. Add `--train-years`/`--val-years` CLI args to `train.py`, `evaluate.py`, `log_tso_baseline.py`, `train_mlforecast.py`
3. Resolution order: CLI arg > dataset config > global settings (fallback)
4. Pass split params explicitly to `run_training()` and other consumers
5. Consolidate temporal_split duplicates to import from `training.harness`
6. Log `split.train_years` and `split.val_years` as MLflow params

## Feature Metadata

**Feature Type**: Refactor
**Estimated Complexity**: Medium
**Primary Systems Affected**: config, training harness, all training/eval scripts
**Dependencies**: None (internal refactor)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `src/windcast/config.py` (lines 10-19, 53-61, 83-93, 158-198) — Dataset config classes + WindCastSettings. train_years/val_years at lines 170-171
- `src/windcast/training/harness.py` (lines 59-80) — Canonical `temporal_split()` function
- `src/windcast/training/harness.py` (lines 153-212) — `run_training()` reads `settings.train_years` at line 212
- `scripts/train.py` (full file, 152 lines) — Unified training CLI, calls `run_training()`
- `scripts/evaluate.py` (lines 91-106, 251) — Duplicate `_temporal_split()`, reads settings
- `scripts/log_tso_baseline.py` (lines 47-57, 102-105) — Inline split logic, reads settings
- `scripts/train_mlforecast.py` (lines 43-58, 170, 212-216) — Duplicate `_temporal_split()` with `ds` column
- `tests/training/test_harness.py` (lines 39-62, 213-231) — Tests for temporal_split + run_training mock

### Patterns to Follow

**CLI arg pattern** (from `scripts/train.py`):
```python
parser.add_argument(
    "--horizons",
    type=int,
    nargs="+",
    default=None,
    help="Forecast horizons in steps. Default: from settings",
)
```
Follow same pattern: `default=None`, resolve later with fallback chain.

**Config resolution pattern** (from `scripts/train.py:104-116`):
```python
domain_feature_defaults = {"wind": "wind_baseline", ...}
feature_set = args.feature_set or domain_feature_defaults[domain]
```
Mirror this: `train_years = args.train_years or dataset_cfg.train_years or settings.train_years`

**MLflow param logging** (from `harness.py:247-265`):
```python
params: dict[str, Any] = {
    "split.train_start": str(train_df[ts_col].min()),
    ...
}
```
Add `"split.train_years"` and `"split.val_years"` to this dict.

---

## IMPLEMENTATION PLAN

### Phase 1: Add split fields to dataset configs

Add `train_years` and `val_years` to the 3 dataset config base classes. Set per-dataset defaults that match the data available.

### Phase 2: Update harness to accept explicit split params

Add `train_years`/`val_years` parameters to `run_training()` so callers pass them explicitly instead of the harness reading from global settings.

### Phase 3: Add CLI args to all scripts

Add `--train-years`/`--val-years` to `train.py`, `evaluate.py`, `log_tso_baseline.py`, `train_mlforecast.py`. Resolution: CLI > dataset config > global settings.

### Phase 4: Consolidate temporal_split duplicates

Replace the 3 duplicate `_temporal_split()` functions with imports from `training.harness`. Handle the `ds` vs `timestamp_utc` column variance via a parameter.

### Phase 5: Tests + validation

Update existing tests, verify all scripts work, run full validation.

---

## STEP-BY-STEP TASKS

### Task 1: UPDATE `src/windcast/config.py` — Add split fields to dataset configs

- **IMPLEMENT**: Add `train_years: int` and `val_years: int` fields to `DatasetConfig`, `DemandDatasetConfig`, and `SolarDatasetConfig`
- **VALUES**:
  - `DatasetConfig` (wind): default `train_years=5, val_years=1` (Kelmarsh has ~8y data)
  - `HILL_OF_TOWIE`: `train_years=5, val_years=1`
  - `PENMANSHIEL`: `train_years=5, val_years=1`
  - `DemandDatasetConfig`: default `train_years=5, val_years=1`
  - `SPAIN_DEMAND`: `train_years=2, val_years=1` (only 4y of data: 2015-2018)
  - `RTE_FRANCE`: `train_years=8, val_years=2` (11y of data: 2014-2024)
  - `SolarDatasetConfig`: default `train_years=5, val_years=1`
  - `PVDAQ_SYSTEM4`: `train_years=5, val_years=1`
- **KEEP**: `WindCastSettings.train_years` and `val_years` stay as ultimate fallback (do NOT remove)
- **VALIDATE**: `uv run pyright src/windcast/config.py`

### Task 2: UPDATE `src/windcast/training/harness.py` — Accept split params in temporal_split and run_training

- **IMPLEMENT**: Add optional `timestamp_col` parameter to `temporal_split()`:
  ```python
  def temporal_split(
      df: pl.DataFrame,
      train_years: int,
      val_years: int,
      timestamp_col: str = "timestamp_utc",
  ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
  ```
  Replace hardcoded `"timestamp_utc"` with `timestamp_col` in the function body.

- **IMPLEMENT**: Add `train_years` and `val_years` parameters to `run_training()`:
  ```python
  def run_training(
      ...,
      train_years: int | None = None,
      val_years: int | None = None,
  ) -> None:
  ```
  Resolution inside `run_training()`:
  ```python
  effective_train_years = train_years if train_years is not None else settings.train_years
  effective_val_years = val_years if val_years is not None else settings.val_years
  ```
  Use `effective_train_years`/`effective_val_years` in the `temporal_split()` call (line 212) and log them as MLflow params.

- **ADD**: Log year counts as MLflow params alongside existing boundary timestamps:
  ```python
  "split.train_years": effective_train_years,
  "split.val_years": effective_val_years,
  ```
- **VALIDATE**: `uv run pyright src/windcast/training/harness.py`

### Task 3: UPDATE `scripts/train.py` — Add CLI args and resolve split config

- **ADD** CLI args after the existing `--horizons` arg:
  ```python
  parser.add_argument("--train-years", type=int, default=None,
      help="Training split in years. Default: from dataset config")
  parser.add_argument("--val-years", type=int, default=None,
      help="Validation split in years. Default: from dataset config")
  ```
- **IMPLEMENT** resolution chain before calling `run_training()`:
  ```python
  from windcast.config import DATASETS
  dataset_cfg = DATASETS[dataset]
  resolved_train_years = args.train_years or getattr(dataset_cfg, "train_years", settings.train_years)
  resolved_val_years = args.val_years or getattr(dataset_cfg, "val_years", settings.val_years)
  ```
- **PASS** to `run_training()`:
  ```python
  run_training(
      ...,
      train_years=resolved_train_years,
      val_years=resolved_val_years,
  )
  ```
- **VALIDATE**: `uv run python scripts/train.py --help` (check new args appear)

### Task 4: UPDATE `scripts/evaluate.py` — Replace duplicate temporal_split, add CLI args

- **REMOVE**: Local `_temporal_split()` function (lines 91-106)
- **ADD** import: `from windcast.training.harness import temporal_split`
- **ADD** CLI args `--train-years` and `--val-years` (same pattern as train.py)
- **UPDATE** line 251: Use imported `temporal_split()` with resolved split params
- **IMPLEMENT** resolution chain (same as train.py: CLI > dataset config > settings)
- **VALIDATE**: `uv run python scripts/evaluate.py --help`

### Task 5: UPDATE `scripts/log_tso_baseline.py` — Replace inline split logic, add CLI args

- **ADD** import: `from windcast.training.harness import temporal_split`
- **REMOVE**: Inline split logic (lines 54-57) — replace with:
  ```python
  dataset_cfg = DATASETS["rte_france"]
  effective_train_years = args.train_years or dataset_cfg.train_years
  effective_val_years = args.val_years or dataset_cfg.val_years
  train_df, val_df, test_df = temporal_split(df, effective_train_years, effective_val_years)
  ```
- **ADD** CLI args `--train-years` and `--val-years`
- **UPDATE** MLflow param logging to include `split.train_years` and `split.val_years`
- **VALIDATE**: `uv run python scripts/log_tso_baseline.py --help`

### Task 6: UPDATE `scripts/train_mlforecast.py` — Replace duplicate temporal_split, add CLI args

- **REMOVE**: Local `_temporal_split()` function (lines 43-58)
- **ADD** import: `from windcast.training.harness import temporal_split`
- **UPDATE** line 170: Use imported `temporal_split()` with `timestamp_col="ds"`
- **ADD** CLI args `--train-years` and `--val-years`
- **IMPLEMENT** resolution chain
- **VALIDATE**: `uv run python scripts/train_mlforecast.py --help`

### Task 7: UPDATE tests

- **UPDATE** `tests/training/test_harness.py`:
  - Add test for `temporal_split()` with custom `timestamp_col`
  - Update `test_run_training_mock_backend` to pass `train_years`/`val_years` explicitly
  - Test that `run_training()` uses passed values over settings
- **VALIDATE**: `uv run pytest tests/training/test_harness.py -v`

### Task 8: Full validation

- **VALIDATE**: `uv run ruff check src/ tests/ scripts/`
- **VALIDATE**: `uv run ruff format --check src/ tests/ scripts/`
- **VALIDATE**: `uv run pyright src/`
- **VALIDATE**: `uv run pytest tests/ -v`

---

## TESTING STRATEGY

### Unit Tests

- `temporal_split()` with `timestamp_col="ds"` (new param)
- `run_training()` with explicit `train_years`/`val_years` params
- Resolution chain: explicit > dataset_config > settings fallback

### Integration Tests

- Existing `test_run_training_mock_backend` updated to pass split params
- Verify MLflow logs `split.train_years` and `split.val_years` params

### Edge Cases

- `train_years=None` falls back to dataset config, then to settings
- `val_years=0` should still work (all data = train + test, no val)
- Dataset config with no `train_years` field (backward compat with getattr)

---

## VALIDATION COMMANDS

```bash
# Level 1: Lint + types
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/

# Level 2: Tests
uv run pytest tests/ -v

# Level 3: CLI smoke tests
uv run python scripts/train.py --help          # --train-years/--val-years visible
uv run python scripts/evaluate.py --help
uv run python scripts/log_tso_baseline.py --help
uv run python scripts/train_mlforecast.py --help
```

---

## ACCEPTANCE CRITERIA

- [ ] `uv run python scripts/train.py --domain demand --dataset rte_france` uses 8/2 split without env vars
- [ ] `uv run python scripts/train.py` (wind/kelmarsh) uses 5/1 split by default
- [ ] `uv run python scripts/train.py --train-years 3 --val-years 1` overrides dataset config
- [ ] MLflow runs log `split.train_years` and `split.val_years` as params
- [ ] No duplicate `temporal_split()` functions remain (only one in `training.harness`)
- [ ] All 4 scripts accept `--train-years`/`--val-years` CLI args
- [ ] `ruff check`, `ruff format`, `pyright`, `pytest` all pass
- [ ] No behavioral change for existing wind pipeline (Kelmarsh defaults preserved)

---

## NOTES

- `WindCastSettings.train_years`/`val_years` kept as ultimate fallback — no breaking change
- `build_features.py` does NOT use temporal split, so no changes needed there
- The `getattr(dataset_cfg, "train_years", settings.train_years)` pattern handles any future dataset config that might not have split fields (defensive)
- After this refactor, the `WINDCAST_TRAIN_YEARS`/`WINDCAST_VAL_YEARS` env vars still work but are no longer necessary for demand runs
- The explicit `WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2 uv run ...` permission entries in settings.local.json can be replaced by a single `Bash(uv run:*)` wildcard
