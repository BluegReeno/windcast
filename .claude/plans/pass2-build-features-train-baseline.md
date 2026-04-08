# Feature: Pass 2 — Build Features + Train Baseline on Kelmarsh Real Data

## Feature Description

Run the wind feature engineering pipeline on Kelmarsh real SCADA data (6 turbines, 473k rows each), then train XGBoost baseline models per forecast horizon with MLflow tracking. This is the first time the code runs on real data — expect data-shape surprises.

## User Story

As an energy ML engineer,
I want to run the full feature-engineering + training pipeline on real Kelmarsh data,
So that I get actual metrics in MLflow to validate the framework and populate the WN presentation.

## Problem Statement

The framework code (features, training, evaluation) has been developed and tested with synthetic data (234 tests passing). It has never been run on the real Kelmarsh processed Parquets (produced in Pass 1). We need real metrics — MAE, RMSE, skill score — to validate the approach and populate presentation slides.

## Solution Statement

1. Run `build_features.py` on all 6 Kelmarsh turbine Parquets with `wind_baseline` feature set
2. Run `train.py` on one turbine (kwf1) with `wind_baseline` to get MLflow results
3. Verify skill score > 0 at short horizons (h1, h6) — model beats persistence
4. Fix any issues that arise from real data shapes/values

## Feature Metadata

**Feature Type**: Execution / Validation (no new code expected, just running existing pipeline)
**Estimated Complexity**: Low-Medium (code exists, but real data may surface bugs)
**Primary Systems Affected**: `features/wind.py`, `scripts/build_features.py`, `scripts/train.py`, MLflow
**Dependencies**: Processed Parquets in `data/processed/kelmarsh_kwf*.parquet` (DONE)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `scripts/build_features.py` — Feature build CLI. Reads from `data/processed/`, writes to `data/features/`. Uses `--feature-set wind_baseline` by default for wind domain.
- `scripts/train.py` — Training CLI. Reads from `data/features/`, trains XGBoost per horizon, logs to MLflow. Default: `--turbine-id kwf1 --feature-set wind_baseline`.
- `src/windcast/features/wind.py` — Feature engineering: QC filter (qc_flag==0), lags [1,2,3,6,12,24], rolling [6,12,24], cyclic wind direction.
- `src/windcast/features/registry.py` (lines 15-30) — `wind_baseline` feature set: 11 columns.
- `src/windcast/models/xgboost_model.py` — XGBoost wrapper. Config: 500 trees, lr=0.05, depth=6, early_stopping=50.
- `src/windcast/models/evaluation.py` — Metrics: MAE, RMSE, bias, MAPE, skill_score.
- `src/windcast/tracking/mlflow_utils.py` — MLflow logging helpers.
- `src/windcast/config.py` (lines 140-179) — Settings: train_years=5, val_years=1, test_years=1, horizons=[1,6,12,24,48].

### No New Files to Create

This is an execution pass. All code exists. We may need to fix bugs discovered during execution.

### Data Context

Real Kelmarsh kwf1 Parquet:
- **Shape**: 473,184 rows x 15 columns
- **Date range**: 2016-01-03 to 2024-12-31 (9 years)
- **QC distribution**: 311,339 OK (66%), 110,785 suspect (23%), 51,060 bad (11%)
- **After QC filter**: ~311k rows per turbine available for features
- **Resolution**: 10-minute intervals
- **Temporal split** (5+1+1 years): train ~2016-2021 (~166k rows after QC), val ~2021-2022 (~33k), test ~2022-2024 (~remainder)

### Key Wind Baseline Features (11 columns)

```
wind_speed_ms, wind_dir_sin, wind_dir_cos,
active_power_kw_lag1, _lag2, _lag3, _lag6,
active_power_kw_roll_mean_6, _roll_mean_12, _roll_mean_24,
active_power_kw_roll_std_6
```

---

## IMPLEMENTATION PLAN

### Phase 1: Build Features (all 6 turbines)

Run `build_features.py` with default wind_baseline. Expected outputs: 6 Parquet files in `data/features/`.

**Potential issues:**
- `drop_nulls()` in build_features.py will remove rows with null lags/rolling at series start. With lag24 = 24 steps x 10 min = 4 hours of data lost. Acceptable.
- QC filter drops 34% of data (qc_flag != 0). If QC gaps are clustered, lags may have many nulls → more rows dropped. Watch the row counts in logs.

### Phase 2: Train Baseline (kwf1)

Run `train.py` with defaults (kwf1, wind_baseline, horizons [1,6,12,24,48]).

**Potential issues:**
- Temporal split: 5+1+1=7 years, data has 9 years. Test set will be 2 years (2022-2024). Fine.
- XGBoost accepts Polars DataFrames directly (>=2.0). Should work.
- `model.predict(X_val)` returns numpy array. `compute_metrics` expects numpy. OK.
- Persistence baseline uses `active_power_kw_lag1` from feature columns. This is in wind_baseline. OK.
- Horizons in 10-min steps: h1=10min, h6=1h, h12=2h, h24=4h, h48=8h.

### Phase 3: Verify Results

Check MLflow for:
- 5 nested runs (one per horizon) under parent run `kwf1-wind_baseline`
- Skill score > 0 at h1 (10 min) — should be easy for XGBoost vs persistence
- Skill score degradation at longer horizons — expected
- MAE in reasonable range (Kelmarsh rated=2050 kW, expect MAE 100-400 kW depending on horizon)

---

## STEP-BY-STEP TASKS

### Task 1: RUN build_features.py on all Kelmarsh turbines

```bash
uv run python scripts/build_features.py --feature-set wind_baseline
```

**VALIDATE:**
- Check 6 Parquet files created in `data/features/`
- Each file should have ~300k+ rows (311k minus nulls from lags)
- Each file should have 15 original columns + 11 feature columns = ~26+ columns
- Log output should show "Feature set: wind_baseline, output: data/features"

```bash
ls -la data/features/kelmarsh_*.parquet
uv run python -c "
import polars as pl
df = pl.read_parquet('data/features/kelmarsh_kwf1.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns}')
print(f'Date range: {df[\"timestamp_utc\"].min()} to {df[\"timestamp_utc\"].max()}')
print(f'Null count per feature col:')
for c in ['wind_speed_ms','wind_dir_sin','active_power_kw_lag1','active_power_kw_roll_mean_24']:
    print(f'  {c}: {df[c].null_count()}')
"
```

### Task 2: RUN train.py on kwf1 with wind_baseline

```bash
uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_baseline
```

**VALIDATE:**
- Log output shows temporal split sizes (train ~160-180k, val ~30-40k, test not used in training)
- 5 horizons trained: h1, h6, h12, h24, h48
- Each horizon logs MAE, RMSE, skill_score
- No errors or warnings about missing columns

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('enercast-kelmarsh')
if exp:
    runs = client.search_runs(exp.experiment_id, order_by=['start_time DESC'], max_results=10)
    for r in runs:
        print(f'{r.info.run_name}: {dict(list(r.data.metrics.items())[:5])}')
else:
    print('Experiment not found')
"
```

### Task 3: VERIFY results — skill score > 0

Expected results (order of magnitude):
- h1 (10 min): skill_score 0.1-0.5 (easy to beat persistence at short horizon with wind speed features)
- h6 (1 hour): skill_score 0.05-0.3
- h12 (2 hours): skill_score ~0.0-0.2 (may be close to persistence)
- h24 (4 hours): skill_score may be near 0 or negative
- h48 (8 hours): skill_score likely negative without NWP features

**If skill_score < 0 at h1**: Something is wrong. Debug: check if lag1 column matches what persistence uses, check if temporal split is correct (no leakage but also no gap).

**If all skill_scores are negative**: The model is worse than persistence at all horizons. Possible causes:
- Feature engineering bug (wrong lag computation)
- Temporal split leaves insufficient training data
- XGBoost overfitting (check best_iteration vs n_estimators)

### Task 4: UPDATE STATUS.md

Mark Pass 2 as done:
- `[x]` for Pass 2 task
- Note actual metrics obtained
- Update Wednesday exit criteria checkboxes

---

## TESTING STRATEGY

No new tests needed. This is an execution pass on real data. If bugs are found, fix and re-run.

**Existing tests still valid:**
```bash
uv run pytest tests/ -v --tb=short
```

---

## VALIDATION COMMANDS

### Level 1: Pre-flight (ensure code is clean)

```bash
uv run ruff check src/ scripts/
uv run pyright src/
```

### Level 2: Feature Build

```bash
uv run python scripts/build_features.py --feature-set wind_baseline
```

Expected: 6 Parquet files in `data/features/`, ~300k rows each, no errors.

### Level 3: Training

```bash
uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_baseline
```

Expected: MLflow run with 5 nested horizon runs, metrics logged.

### Level 4: MLflow Verification

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('enercast-kelmarsh')
runs = client.search_runs(exp.experiment_id)
print(f'Total runs: {len(runs)}')
for r in runs:
    if 'h1_skill_score' in r.data.metrics:
        print(f'h1 skill: {r.data.metrics[\"h1_skill_score\"]:.3f}')
"
```

---

## ACCEPTANCE CRITERIA

- [ ] `data/features/` contains 6 Kelmarsh feature Parquets
- [ ] Each feature Parquet has ~300k+ rows and 26+ columns
- [ ] MLflow `enercast-kelmarsh` experiment exists with runs
- [ ] At least h1 skill_score > 0 (model beats persistence at 10 min)
- [ ] No ruff/pyright errors introduced
- [ ] STATUS.md updated with Pass 2 results

---

## COMPLETION CHECKLIST

- [ ] build_features.py ran successfully on all 6 turbines
- [ ] train.py ran successfully on kwf1 with wind_baseline
- [ ] MLflow shows metrics for all 5 horizons
- [ ] Skill score > 0 verified at h1
- [ ] Actual metrics noted for presentation use
- [ ] STATUS.md updated

---

## NOTES

### Time Budget
This is Pass 2 of a 10-pass sprint. Target: ~20-30 minutes. If build_features or train takes too long, run on kwf1 only first, then parallelize.

### Horizon Interpretation
The horizons [1,6,12,24,48] are in **steps of the data resolution** (10 min for Kelmarsh):
- h1 = 10 min ahead
- h6 = 1 hour ahead  
- h12 = 2 hours ahead
- h24 = 4 hours ahead
- h48 = 8 hours ahead

This is NOT hours ahead. The `train.py` shift(-h) creates the correct target.

### Expected Skill Score Pattern
- Short horizons (h1-h6): XGBoost should beat persistence because wind speed is a strong predictor
- Long horizons (h24-h48): Without NWP forecasts, XGBoost relies on stale lags → may not beat persistence
- This is the expected result and motivates the "enriched" and "full" feature sets in Pass 3

### Data Volume
- 6 turbines x 311k QC-OK rows x 10 min = manageable in memory
- XGBoost hist method handles ~300k rows easily
- Feature build: mostly Polars expressions, should be fast (<30s per turbine)
- Training: 5 horizons x 500 trees with early stopping → ~1-3 min per horizon

### Confidence Score: 8/10
High confidence because:
- All code is tested (234 tests pass)
- Data format matches schema (verified in Pass 1)
- Standard XGBoost pipeline on tabular data
- Minor risks: Polars version compatibility, XGBoost Polars input, null handling at edges
