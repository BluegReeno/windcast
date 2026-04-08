# Feature: Experiment Lineage Schema — Complete ML Provenance

## Feature Description

Add a structured lineage schema to the framework so every MLflow run captures the full provenance chain: data source → QC → features → split → model → evaluation. This is the metadata layer that makes EnerCast a **framework** rather than a collection of scripts.

## User Story

As an energy ML engineer,
I want every training run to automatically capture full provenance (data version, QC stats, split boundaries, feature set, hyperparameters, git commit),
So that I can reproduce any result, compare runs fairly, and trace regressions to their root cause.

## Problem Statement

Today the framework logs metrics and hyperparameters, but NOT:
- Data provenance (which file, how many rows, date range, QC stats)
- Split boundaries (exact dates for train/val/test)
- Run purpose/stage (dev vs prod, experiment vs baseline)
- Git commit (auto-captured by MLflow but not verified)
- Data resolution (10min for wind, 1h for demand)
- Feature engineering version

This makes it impossible to answer: "Why did run B score differently from run A?" — was it different data? Different split? Different QC? Different features?

## Solution Statement

1. Define a **lineage schema** as a Pydantic model (`RunLineage`)
2. Add a **`log_run_lineage()`** helper that logs the full schema to MLflow (tags + params + datasets)
3. Use **MLflow tags** for searchable metadata (stage, purpose, domain)
4. Use **`mlflow.data.from_polars()`** for native data provenance with auto-hash
5. Define **naming conventions** as code constants
6. Update `train.py` and `train_mlforecast.py` to call `log_run_lineage()`

## Feature Metadata

**Feature Type**: Enhancement
**Estimated Complexity**: Medium
**Primary Systems Affected**: `src/windcast/tracking/`, `scripts/train.py`, `scripts/train_mlforecast.py`
**Dependencies**: MLflow 3.10+ (already installed)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `src/windcast/tracking/mlflow_utils.py` — Current MLflow helpers. 4 functions: setup_mlflow, log_feature_set, log_evaluation_results, log_dataframe_artifact. **This is where we add the new lineage helpers.**
- `src/windcast/tracking/__init__.py` — Package exports. Must add new exports.
- `scripts/train.py` (lines 186-245) — Parent/nested run structure. Currently logs: domain, dataset, horizons, n_train/n_val/n_test, feature_set, turbine_id. **Missing: split dates, data hash, QC stats, resolution, stage tag.**
- `scripts/train_mlforecast.py` (lines 184-223) — Same pattern as train.py. Same gaps.
- `scripts/evaluate.py` (lines 264-336) — Eval run logging. Same gaps.
- `src/windcast/config.py` (lines 140-179) — WindCastSettings: train_years=5, val_years=1, forecast_horizons=[1,6,12,24,48], mlflow_tracking_uri.
- `src/windcast/features/registry.py` (lines 6-12) — FeatureSet dataclass: name, columns, description.
- `src/windcast/models/xgboost_model.py` (lines 14-26) — XGBoostConfig Pydantic model with all hyperparams.

### New Files to Create

- `src/windcast/tracking/lineage.py` — RunLineage model + log_run_lineage() helper

### Patterns to Follow

**Logging pattern** (from xgboost_model.py:68-71):
```python
if mlflow.active_run():
    mlflow.log_params(config.model_dump())
    mlflow.log_metric("best_iteration", model.best_iteration)
```

**Config as Pydantic model** (from xgboost_model.py:14-26):
```python
class XGBoostConfig(BaseModel):
    objective: str = "reg:squarederror"
    n_estimators: int = 500
    ...
```

**Feature set logging** (from mlflow_utils.py:31-43):
```python
mlflow.log_param("feature_set", feature_set_name)
mlflow.log_dict({"feature_set": feature_set_name, "columns": feature_columns}, "feature_set.json")
```

---

## LINEAGE SCHEMA DESIGN

### What goes WHERE in MLflow

| MLflow Concept | What to store | Why here |
|----------------|---------------|----------|
| **Tags** | stage, purpose, domain, backend, git_branch | Searchable, mutable, categorical |
| **Params** | split dates, data resolution, n_rows_*, QC stats, feature_set | Immutable run config, filterable |
| **Metrics** | MAE, RMSE, skill_score (already done) | Numeric, comparable |
| **Datasets** | `mlflow.data.from_polars()` with source path | Native provenance with auto-hash |
| **Artifacts** | feature_set.json (already done), lineage.json (full schema dump) | Rich structured data |
| **Notes** | `mlflow.note.content` tag — human-readable run description | Markdown, visible in UI |

### Tag Schema (searchable metadata)

```python
REQUIRED_TAGS = {
    "enercast.stage": "dev|test|prod",         # WattCast lesson: distinguish test from prod
    "enercast.purpose": "baseline|experiment|sweep|eval",
    "enercast.domain": "wind|demand|solar",
    "enercast.backend": "xgboost|mlforecast",
    "enercast.data_resolution_min": "10|60",    # Critical for horizon interpretation
    "mlflow.source.git.branch": "<auto>",       # Must set manually (not auto outside Databricks)
}
# mlflow.source.git.commit is auto-captured when running scripts
```

### Param Schema (immutable run config)

```python
# Data provenance (NEW)
"data.source_file": "data/features/kelmarsh_kwf1.parquet",
"data.n_rows_raw": 473184,          # Before QC
"data.n_rows_after_qc": 311339,     # After QC filter
"data.n_rows_features": 285862,     # After feature engineering (lag null drop)
"data.date_start": "2016-01-03",
"data.date_end": "2024-12-31",

# Split boundaries (NEW)
"split.train_start": "2016-05-03",
"split.train_end": "2021-05-03",
"split.val_start": "2021-05-03",
"split.val_end": "2022-05-03",
"split.test_start": "2022-05-03",

# QC summary (NEW)
"qc.ok_pct": 65.8,
"qc.suspect_pct": 23.4,
"qc.bad_pct": 10.8,

# Already logged (keep as-is)
"domain", "dataset", "turbine_id", "horizons",
"n_train", "n_val", "n_test",
"feature_set", "n_features",
```

### Dataset Tracking (native MLflow)

```python
import mlflow.data

train_dataset = mlflow.data.from_polars(
    train_df,
    source=str(parquet_path),
    name=f"{dataset}-{turbine_id}-train",
    targets=target_col,
)
mlflow.log_input(train_dataset, context="training")
```

This gives us: auto-computed data hash (digest), schema, source path, row count — all visible in MLflow UI.

### Naming Conventions (as code constants)

```python
# Experiment naming
# Pattern: enercast-{dataset}[-{backend}]
EXPERIMENT_PATTERN = "enercast-{dataset}"           # XGBoost
EXPERIMENT_MLFORECAST_PATTERN = "enercast-mlforecast-{dataset}"  # MLforecast
# Already follows this pattern — just document it

# Run naming
# Pattern: {entity_id}-{feature_set}
# entity_id = turbine_id (wind) | dataset (demand/solar)
RUN_NAME_PATTERN = "{entity_id}-{feature_set}"
RUN_NAME_MLFORECAST_PATTERN = "mlforecast-{entity_id}-{feature_set}"

# Nested run naming
# Pattern: h{horizon:02d}
HORIZON_RUN_PATTERN = "h{horizon:02d}"

# Model registry naming (future)
# Pattern: {domain}/{dataset}/{target}
MODEL_NAME_PATTERN = "{domain}/{dataset}/{target}"
```

---

## IMPLEMENTATION PLAN

### Phase 1: Lineage Model + Helpers

Create `src/windcast/tracking/lineage.py` with:
1. `RunLineage` Pydantic model — all the metadata fields
2. `log_run_lineage(lineage: RunLineage)` — logs tags + params + datasets + lineage.json artifact
3. `log_data_provenance(df, source_path, name, target_col, context)` — wraps `mlflow.data.from_polars()`
4. Constants for naming conventions and required tags

### Phase 2: Update train.py

Replace the scattered `mlflow.log_params({...})` block (lines 188-200) with:
1. Compute split boundaries (dates) after temporal split
2. Build `RunLineage` from available data
3. Call `log_run_lineage(lineage)`
4. Call `log_data_provenance()` for train and val DataFrames

### Phase 3: Update train_mlforecast.py

Same changes as Phase 2, adapted for mlforecast's logging pattern.

### Phase 4: Update evaluate.py

Add lineage tags to evaluation runs (stage, purpose).

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/windcast/tracking/lineage.py`

**IMPLEMENT** the lineage module:

```python
"""Experiment lineage schema — structured ML provenance."""

import subprocess
import logging
from datetime import datetime

import mlflow
import polars as pl
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# --- Naming Convention Constants ---

EXPERIMENT_PATTERN = "enercast-{dataset}"
EXPERIMENT_MLFORECAST_PATTERN = "enercast-mlforecast-{dataset}"
RUN_NAME_PATTERN = "{entity_id}-{feature_set}"
RUN_NAME_MLFORECAST_PATTERN = "mlforecast-{entity_id}-{feature_set}"
HORIZON_RUN_PATTERN = "h{horizon:02d}"


class SplitInfo(BaseModel):
    """Temporal split boundary dates and row counts."""
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    n_train: int
    n_val: int
    n_test: int


class DataProvenance(BaseModel):
    """Data source metadata."""
    source_file: str
    n_rows_features: int
    date_start: str
    date_end: str
    data_resolution_min: int = 10


class RunLineage(BaseModel):
    """Complete ML run provenance — logged to MLflow as tags, params, and artifact."""
    # Searchable tags
    stage: str = "dev"                    # dev | test | prod
    purpose: str = "experiment"           # baseline | experiment | sweep | eval
    domain: str = "wind"                  # wind | demand | solar
    backend: str = "xgboost"             # xgboost | mlforecast

    # Data provenance
    data: DataProvenance

    # Split info
    split: SplitInfo

    # Feature set
    feature_set: str
    n_features: int

    # Entity (turbine, zone, system)
    entity_id: str                        # kwf1, spain_demand, pvdaq_system4
    dataset: str                          # kelmarsh, spain_demand, pvdaq_system4


def _get_git_branch() -> str:
    """Get current git branch name."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def log_run_lineage(lineage: RunLineage) -> None:
    """Log full lineage to active MLflow run: tags + params + JSON artifact."""
    # Tags (searchable, categorical)
    mlflow.set_tags({
        "enercast.stage": lineage.stage,
        "enercast.purpose": lineage.purpose,
        "enercast.domain": lineage.domain,
        "enercast.backend": lineage.backend,
        "enercast.data_resolution_min": str(lineage.data.data_resolution_min),
        "mlflow.source.git.branch": _get_git_branch(),
    })

    # Params (immutable config)
    mlflow.log_params({
        "data.source_file": lineage.data.source_file,
        "data.n_rows_features": lineage.data.n_rows_features,
        "data.date_start": lineage.data.date_start,
        "data.date_end": lineage.data.date_end,
        "split.train_start": lineage.split.train_start,
        "split.train_end": lineage.split.train_end,
        "split.val_start": lineage.split.val_start,
        "split.val_end": lineage.split.val_end,
        "split.test_start": lineage.split.test_start,
    })

    # Full lineage as JSON artifact
    mlflow.log_dict(lineage.model_dump(), "lineage.json")
    logger.info("Logged run lineage: stage=%s, purpose=%s", lineage.stage, lineage.purpose)


def log_data_provenance(
    df: pl.DataFrame,
    source_path: str,
    name: str,
    target_col: str,
    context: str = "training",
) -> None:
    """Log Polars DataFrame as MLflow dataset with auto-hash."""
    dataset = mlflow.data.from_polars(
        df,
        source=source_path,
        name=name,
        targets=target_col,
    )
    mlflow.log_input(dataset, context=context)
    logger.info("Logged dataset: %s (%s, %d rows)", name, context, len(df))
```

**VALIDATE:**
```bash
uv run ruff check src/windcast/tracking/lineage.py
uv run pyright src/windcast/tracking/lineage.py
```

### Task 2: UPDATE `src/windcast/tracking/__init__.py`

**ADD** new exports:
```python
from windcast.tracking.lineage import (
    RunLineage,
    DataProvenance,
    SplitInfo,
    log_run_lineage,
    log_data_provenance,
)
```

**VALIDATE:**
```bash
uv run python -c "from windcast.tracking import RunLineage, log_run_lineage, log_data_provenance"
```

### Task 3: UPDATE `scripts/train.py` — add lineage logging

**MODIFY** the `main()` function after temporal split (line ~167):

1. Extract split boundary dates from train_df, val_df, test_df
2. Build `RunLineage` and `DataProvenance` from available data
3. Call `log_run_lineage()` inside the `with mlflow.start_run(...)` block
4. Call `log_data_provenance()` for train and val DataFrames
5. Keep existing `log_feature_set()` call — it logs feature column details

**PATTERN** (after line 170, inside the mlflow.start_run block):

```python
from windcast.tracking import RunLineage, DataProvenance, SplitInfo, log_run_lineage, log_data_provenance

# After temporal split, compute boundaries
ts_col = "timestamp_utc"
lineage = RunLineage(
    stage="dev",
    purpose="baseline",
    domain=domain,
    backend="xgboost",
    data=DataProvenance(
        source_file=str(parquet_path),
        n_rows_features=len(df),
        date_start=str(df[ts_col].min()),
        date_end=str(df[ts_col].max()),
        data_resolution_min=10 if domain == "wind" else 60,
    ),
    split=SplitInfo(
        train_start=str(train_df[ts_col].min()),
        train_end=str(train_df[ts_col].max()),
        val_start=str(val_df[ts_col].min()),
        val_end=str(val_df[ts_col].max()),
        test_start=str(test_df[ts_col].min()),
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
    ),
    feature_set=feature_set,
    n_features=len(available_cols),
    entity_id=run_label,
    dataset=dataset,
)
log_run_lineage(lineage)
log_data_provenance(train_df, str(parquet_path), f"{dataset}-{run_label}-train", dcfg["target"], "training")
log_data_provenance(val_df, str(parquet_path), f"{dataset}-{run_label}-val", dcfg["target"], "validation")
```

**REMOVE** the duplicate params that are now in lineage: `domain`, `dataset`, `horizons`, `n_train`, `n_val`, `n_test`, `turbine_id`. Keep `log_feature_set()` — it logs the column list which lineage doesn't duplicate.

**GOTCHA**: Don't double-log params that are already in `log_run_lineage()`. MLflow raises on duplicate param keys.

**VALIDATE:**
```bash
uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_baseline 2>&1 | head -20
```

### Task 4: UPDATE `scripts/train_mlforecast.py` — add lineage logging

**Same pattern as Task 3**, adapted:
- `backend="mlforecast"` in RunLineage
- Keep existing mlforecast-specific params (strategy, etc.)

**VALIDATE:**
```bash
uv run ruff check scripts/train_mlforecast.py
```

### Task 5: UPDATE `scripts/evaluate.py` — add lineage tags

**ADD** tags to eval runs:
```python
mlflow.set_tags({
    "enercast.stage": "dev",
    "enercast.purpose": "eval",
    "enercast.domain": domain,
    "mlflow.source.git.branch": _get_git_branch(),
})
```

**VALIDATE:**
```bash
uv run ruff check scripts/evaluate.py
```

### Task 6: ADD tests for lineage module

**CREATE** `tests/tracking/test_lineage.py`:
- Test `RunLineage` model creation with valid data
- Test `log_run_lineage()` logs expected tags and params (mock MLflow)
- Test `log_data_provenance()` calls `mlflow.log_input` (mock MLflow)
- Test `_get_git_branch()` returns a string

**VALIDATE:**
```bash
uv run pytest tests/tracking/test_lineage.py -v
```

---

## TESTING STRATEGY

### Unit Tests
- `RunLineage` model validation (required fields, defaults)
- `log_run_lineage()` with mocked mlflow (verify set_tags, log_params, log_dict called)
- `log_data_provenance()` with mocked mlflow.data.from_polars

### Integration Tests
- Run `train.py` on kwf1, then query MLflow for expected tags and params
- Verify `lineage.json` artifact exists and parses correctly

---

## VALIDATION COMMANDS

### Level 1: Syntax & Types
```bash
uv run ruff check src/windcast/tracking/ scripts/
uv run pyright src/windcast/tracking/
```

### Level 2: Unit Tests
```bash
uv run pytest tests/ -v --tb=short
```

### Level 3: Integration (run training, check MLflow)
```bash
uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_baseline
uv run python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('enercast-kelmarsh')
runs = client.search_runs(exp.experiment_id, order_by=['start_time DESC'], max_results=1)
r = runs[0]
print('Tags:', {k: v for k, v in r.data.tags.items() if k.startswith('enercast.')})
print('Split params:', {k: v for k, v in r.data.params.items() if k.startswith('split.')})
print('Data params:', {k: v for k, v in r.data.params.items() if k.startswith('data.')})
"
```

---

## ACCEPTANCE CRITERIA

- [ ] Every training run logs: stage, purpose, domain, backend, git_branch as MLflow tags
- [ ] Every training run logs: split boundaries (6 dates), data source file, row counts as params
- [ ] Every training run logs: `lineage.json` artifact with full schema
- [ ] Every training run logs: train/val DataFrames as MLflow datasets (with auto-hash)
- [ ] Naming conventions are defined as constants in `lineage.py`
- [ ] No duplicate param keys (remove from train.py what's now in lineage)
- [ ] All existing tests still pass
- [ ] New lineage tests pass
- [ ] ruff + pyright clean

---

## NOTES

### Why tags vs params?

- **Tags** are mutable and searchable in MLflow UI filters. Use for categorical metadata you'd filter on: "show me all prod runs", "show me all wind runs".
- **Params** are immutable. Use for values that define the run: split dates, row counts. You can filter on these too.
- **Metrics** are numeric time-series. Already used correctly for MAE/RMSE/skill.

### Data resolution is critical

`data_resolution_min` as a tag solves the "h6 = 1 hour or 6 hours?" ambiguity. With 10-min data, h6=1h. With hourly demand data, h6=6h. Without this tag, horizon comparison across domains is impossible.

### Stage tag prevents WattCast incident

The `enercast.stage` tag directly addresses the WattCast lesson where a test model overwrote prod. With this tag, you can filter MLflow to only show prod runs, and any model promotion workflow can check the tag before overwriting.

### Confidence Score: 8/10

High confidence because:
- MLflow APIs are well-documented and tested
- No changes to the ML pipeline itself — just metadata logging
- Pydantic model ensures schema validation at creation time
- The main risk is `mlflow.data.from_polars()` — need to verify it works with timezone-aware timestamps
