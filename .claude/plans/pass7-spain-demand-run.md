# Feature: Pass 7 — Run Spain Demand Pipeline End-to-End

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Execute the demand domain end-to-end on real Spain ENTSO-E data to prove the EnerCast framework is domain-agnostic. All demand code (schema, parser, QC, features, scripts) is already implemented, tested (50/50 passing), and wired into the multi-domain CLI. Pass 7 is a **runbook** task: download the dataset, run 4 scripts, verify MLflow contains side-by-side wind + demand experiments, and capture the metrics needed for the WeatherNews presentation.

**Zero core code changes are expected.** If the run exposes a real bug, fix it; otherwise the mission is to produce real numbers and MLflow artifacts.

## User Story

As a WeatherNews evaluator,
I want to see the same pipeline handle Kelmarsh wind and Spain demand data in one MLflow UI,
So that I believe EnerCast's standardization is real across structurally different domains.

## Problem Statement

The framework was built and unit-tested domain-agnostic, but has only been run end-to-end on Kelmarsh. Without real demand metrics:
- The presentation cannot show "same pipeline, 2 domains, real numbers"
- Success criterion #2 in PRD ("Demand pipeline runs through the SAME framework") is unverified
- MLflow UI shows only wind experiments
- Risk that hidden coupling surfaces on first real demand run (different target column, hourly resolution, year-range mismatch)

## Solution Statement

Four-step runbook:
1. **Acquire data** — manual download of `nicholasjhana/energy-consumption-generation-prices-and-weather` from Kaggle (Kaggle CLI not configured locally).
2. **Run ingest script** — `scripts/ingest_spain_demand.py` produces `data/processed/spain_demand.parquet`.
3. **Build features + train for 3 feature sets** (`demand_baseline`, `demand_enriched`, `demand_full`), with a **non-default train/val window** because Spain covers only 4 years (2015-2018), not 6+.
4. **Verify MLflow + export comparison chart** via `scripts/compare_runs.py`. Update STATUS.md with the real metrics table.

**Critical config override:** Spain data spans 2015-01-01 to 2018-12-31 (~4 years). The default `settings.train_years=5` would put the entire dataset in the train split (val/test empty). Override via Pydantic env vars at invocation time:

```bash
export WINDCAST_TRAIN_YEARS=2
export WINDCAST_VAL_YEARS=1
# → train=2015-2016, val=2017, test=2018
```

## Feature Metadata

**Feature Type**: Execution / runbook (no new code expected)
**Estimated Complexity**: Low (code exists) / Medium (real data surprises possible)
**Primary Systems Affected**: `scripts/ingest_spain_demand.py`, `scripts/build_features.py`, `scripts/train.py`, MLflow store
**Dependencies**: Kaggle dataset (manual download), existing `mlflow.db`, existing `uv` environment

---

## CONTEXT REFERENCES

### Relevant Codebase Files — YOU MUST READ THESE BEFORE IMPLEMENTING!

**Scripts (run top-to-bottom, already multi-domain):**
- `scripts/ingest_spain_demand.py` (full, 86 lines) — Parse + QC + single Parquet write. Already uses `settings.raw_dir / "spain" / ...` as default input.
- `scripts/build_features.py` (lines 75-100, 142-180) — Supports `--domain demand`, dispatches to `build_demand_features`, writes `spain_demand_features.parquet`.
- `scripts/train.py` (lines 28-32 `DOMAIN_CONFIG`, 169-196 domain dispatch, 218-222 `_temporal_split` call, 244-246 `data_resolution=60` for demand) — Already handles `--domain demand` and uses `load_mw` / `zone_id` / `load_mw_lag1` for target / group / persistence.
- `scripts/compare_runs.py` — Generates MAE + skill bar charts + Markdown table per horizon per experiment. Run this at the end with `--experiment enercast-spain_demand`.

**Demand domain modules (code-complete, do NOT modify):**
- `src/windcast/data/demand_schema.py` — 11-column schema (`load_mw`, `temperature_c`, `wind_speed_ms`, `humidity_pct`, `price_eur_mwh`, `is_holiday`, `is_dst_transition`, `qc_flag`, ...).
- `src/windcast/data/spain_demand.py` (lines 74-101 `_read_energy_csv`, 104-149 `_read_weather_csv`, 152-158 `_aggregate_weather`) — Parses energy CSV (`total load actual` → `load_mw`, `price day ahead` → `price_eur_mwh`) + weather CSV (Kelvin→Celsius, Barcelona pressure / Valencia wind outlier filter, 5-city mean aggregate), joins on UTC timestamp.
- `src/windcast/data/demand_qc.py` (lines 46-80 `run_demand_qc_pipeline`, 15-31 Spain 2015-2018 holidays, 34-43 DST dates) — 6-stage QC.
- `src/windcast/features/demand.py` (full, 145 lines) — `build_demand_features` dispatches baseline/enriched/full; lag features use `.over("zone_id")`; is_holiday cast to Int8 for XGBoost.
- `src/windcast/features/registry.py` (lines 61-104) — `DEMAND_BASELINE` (10 cols), `DEMAND_ENRICHED` (+6 cols), `DEMAND_FULL` (+6 cols including wind/humidity/price/holiday).
- `src/windcast/config.py` (lines 53-71 `DemandDatasetConfig` + `SPAIN_DEMAND`, 117-125 `DemandQCConfig`, 140-145 `DOMAIN_RESOLUTION[demand]=60`, 156-167 `WindCastSettings` env override via `WINDCAST_` prefix).

**MLflow helpers (already emit demand-compatible output):**
- `src/windcast/tracking/mlflow_utils.py` (lines 21-30 `STEPPED_METRIC_MAP`) — Stepped metrics use `minutes_ahead` as the universal step unit, so `h1=60`, `h48=2880` for hourly demand.

**Reference runs already in MLflow (`sqlite:///mlflow.db`):**
- Experiment `enercast-kelmarsh` — 4 parent runs (XGB baseline / enriched / full + AutoGluon full).
- Experiment `enercast-spain_demand` — **does not exist yet**; `setup_mlflow()` will create it on first write.

### New Files to Create

**None.** All code exists. You may create the following artifacts during the run:
- `data/raw/spain/energy_dataset.csv` (downloaded)
- `data/raw/spain/weather_features.csv` (downloaded)
- `data/processed/spain_demand.parquet` (written by ingest)
- `data/features/spain_demand_features.parquet` (written by build_features; overwritten per feature set — see Gotcha §3)
- `reports/comparison_enercast-spain_demand_mae.png` (from compare_runs.py)
- `reports/comparison_enercast-spain_demand_skill.png` (from compare_runs.py)

### Relevant Documentation

**Dataset source:**
- Kaggle: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
- License: CC0 (public domain)
- Contents: `energy_dataset.csv` (35,064 rows × 29 cols, hourly 2015-2018) + `weather_features.csv` (178,396 rows × 17 cols, 5 cities long format). Total ~8 MB extracted.

**Open-Meteo NWP is NOT required for demand.** The `DEMAND_FULL` feature set references `wind_speed_ms`, `humidity_pct`, `price_eur_mwh` — all sourced from the Spain weather CSV itself, not from `data/weather.db`. Do not pass `--weather-db` to `build_features.py` for demand.

**Confirmed current state:**
- `data/weather.db` has NO Spain coverage (confirmed via `get_coverage('40.4168_-3.7038')` → None). This is fine: demand does not use NWP columns.
- `mlflow.db` currently holds Kelmarsh experiments only.
- All 50 demand unit tests pass (`pytest tests/data/test_*demand* tests/features/test_demand.py`).

### Patterns to Follow

**Running a multi-domain script** (pattern from `scripts/train.py:120-196`):
```bash
uv run python scripts/<script>.py --domain demand [--feature-set demand_xxx] [--dataset spain_demand]
```

**Overriding settings at runtime** (pattern from `config.py:148-154`, `env_prefix="WINDCAST_"`):
```bash
WINDCAST_TRAIN_YEARS=2 WINDCAST_VAL_YEARS=1 uv run python scripts/train.py ...
```

**Comparing runs** (pattern from Pass 6b, `scripts/compare_runs.py`):
```bash
uv run python scripts/compare_runs.py --experiment enercast-spain_demand
```

**Naming Conventions:**
- Experiment name: `enercast-spain_demand` (auto-derived from `--dataset`)
- Parent run name: `spain_demand-demand_baseline`, `spain_demand-demand_enriched`, `spain_demand-demand_full`
- Child run name: `h01`, `h06`, `h12`, `h24`, `h48`

**Error Handling:**
- If a script fails, read the full traceback. Do NOT silently skip.
- If ingest produces 0 rows, check timestamp TZ parsing (`%:z` format) and the `city_name` strip step.
- If train fails with "Insufficient data for temporal split", confirm `WINDCAST_TRAIN_YEARS=2 WINDCAST_VAL_YEARS=1` are exported in the current shell.

---

## IMPLEMENTATION PLAN

### Phase 1: Data Acquisition

Obtain the Spain ENTSO-E CSVs and place them under `data/raw/spain/`.

### Phase 2: Ingest + Schema Validation

Parse CSVs → canonical demand schema → QC → Parquet. Verify row count, QC summary, sane load range.

### Phase 3: Train 3 Feature Sets with MLflow

Build features + train once per feature set (`demand_baseline`, `demand_enriched`, `demand_full`). Each run:
- Builds features (overwrites `spain_demand_features.parquet`)
- Trains 5 horizons (h=1,6,12,24,48 hours ahead)
- Logs parent + 5 children + stepped horizon metrics to MLflow

### Phase 4: Verification + Comparison Artifacts

Query MLflow programmatically to confirm 3 parent runs × 5 children exist with positive skill scores. Generate comparison charts. Update STATUS.md with the real metrics table.

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently verifiable.

---

### Task 1: ACQUIRE Spain dataset

**IMPLEMENT**: Download `energy_dataset.csv` and `weather_features.csv` from Kaggle and extract under `data/raw/spain/`.

- **OPTION A (preferred, requires Kaggle credentials at `~/.kaggle/kaggle.json`)**:
  ```bash
  uv run --with kagglehub python -c "
  import kagglehub, shutil
  from pathlib import Path
  src = Path(kagglehub.dataset_download('nicholasjhana/energy-consumption-generation-prices-and-weather'))
  dst = Path('data/raw/spain')
  dst.mkdir(parents=True, exist_ok=True)
  for f in ['energy_dataset.csv', 'weather_features.csv']:
      shutil.copy(src / f, dst / f)
      print(f'Copied {f}')
  "
  ```
- **OPTION B (fallback, manual)**: Open https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather in a browser, click "Download", unzip, move the two CSVs to `data/raw/spain/`.
- **OPTION C (if user has Kaggle CLI installed separately)**: `kaggle datasets download -d nicholasjhana/energy-consumption-generation-prices-and-weather -p data/raw/spain --unzip`

**PATTERN**: No existing pattern — Kelmarsh was shipped as a local ZIP under `data/KelmarshV4/`.
**GOTCHA**: Kaggle's downloaded folder may contain extra files (generation_dataset.csv etc.); only the two target CSVs matter. Do not check the raw CSVs into git (already gitignored via `data/`).
**VALIDATE**:
```bash
ls -la data/raw/spain/energy_dataset.csv data/raw/spain/weather_features.csv
uv run python -c "
import polars as pl
e = pl.read_csv('data/raw/spain/energy_dataset.csv', infer_schema_length=10000)
w = pl.read_csv('data/raw/spain/weather_features.csv', infer_schema_length=10000)
print(f'energy: {e.shape}, cols={len(e.columns)}')
print(f'weather: {w.shape}, cols={len(w.columns)}')
assert e.shape == (35064, 29), f'Unexpected energy shape: {e.shape}'
assert w.shape[0] == 178396, f'Unexpected weather row count: {w.shape[0]}'
print('OK')
"
```
**Expected output**: `energy: (35064, 29)`, `weather: (178396, 17)`, `OK`.

---

### Task 2: INGEST Spain demand → Parquet

**IMPLEMENT**: Run the ingestion script.

```bash
uv run python scripts/ingest_spain_demand.py
```

**PATTERN**: Mirrors Kelmarsh ingestion from Pass 1 — reads CSVs, validates schema, runs QC pipeline, writes a single Parquet.
**IMPORTS**: No code. The script imports `parse_spain_demand`, `run_demand_qc_pipeline`, `validate_demand_schema`.
**GOTCHA**:
- Expect ~35,000 rows in the final Parquet (inner-join of energy + weather on UTC timestamp; should match energy row count minus any weather gaps).
- QC summary should show `qc_ok_pct > 95%`. If `qc_bad_pct > 5%`, inspect the load range — min load 10,000 MW may be too aggressive for Spain's weekend/holiday troughs; check with `uv run python -c "import polars as pl; df=pl.read_parquet('data/processed/spain_demand.parquet'); print(df['load_mw'].describe())"` before adjusting thresholds.
- `total load actual` has 36 NaN rows in the source — these will propagate as `load_mw=null` and survive the join; the feature builder's `drop_nulls()` handles them downstream. Do NOT drop them at parse time.

**VALIDATE**:
```bash
ls -la data/processed/spain_demand.parquet
uv run python -c "
import polars as pl
df = pl.read_parquet('data/processed/spain_demand.parquet')
print('Rows:', len(df))
print('Columns:', df.columns)
print('Date range:', df['timestamp_utc'].min(), '→', df['timestamp_utc'].max())
print('Load MW range:', df['load_mw'].min(), '→', df['load_mw'].max())
print('QC distribution:', df['qc_flag'].value_counts().sort('qc_flag').to_dicts())
print('Holidays count:', df['is_holiday'].sum())
assert len(df) > 30000, 'Too few rows after join'
assert df['timestamp_utc'].min().year == 2015
assert df['timestamp_utc'].max().year == 2018
"
```
**Expected**: ~35k rows, 11 columns matching `DEMAND_SCHEMA`, load range roughly 18,000–41,000 MW, qc_ok > 95%, holidays count > 100.

---

### Task 3: BUILD features + TRAIN `demand_baseline`

**IMPLEMENT**: Build baseline features and train XGBoost on 5 horizons. Override year split for 4-year dataset.

```bash
export WINDCAST_TRAIN_YEARS=2 WINDCAST_VAL_YEARS=1
uv run python scripts/build_features.py --domain demand --feature-set demand_baseline
uv run python scripts/train.py --domain demand --dataset spain_demand --feature-set demand_baseline
```

**PATTERN**: Mirrors Pass 2 / Pass 3 wind training runs — parent run with 5 nested children, stepped metrics replayed on parent via `log_stepped_horizon_metrics`.
**IMPORTS**: Scripts handle imports. `DOMAIN_CONFIG["demand"]` (train.py:30) sets target/group/lag1 correctly.
**GOTCHA**:
- `WINDCAST_TRAIN_YEARS` / `WINDCAST_VAL_YEARS` MUST be exported BEFORE running train.py; `get_settings()` is `lru_cache`'d (config.py:186-188) and reads env vars once at first call. If you forget, train will consume the whole dataset as "train" and log "Insufficient data for temporal split".
- `forecast_horizons=[1,6,12,24,48]` default is correct for hourly demand → horizons mean 1h/6h/12h/24h/48h ahead (stepped metric step = 60/360/720/1440/2880 minutes).
- Data resolution is set to 60 min automatically via `data_resolution = 10 if domain == "wind" else 60` (train.py:246). Solar (15 min) is misrepresented here but not in scope for Pass 7.
- `build_features.py` always writes to the same `spain_demand_features.parquet` — so each feature set run overwrites the previous. That's fine because `train.py` reads the Parquet eagerly before the next build overwrites it. Do NOT parallelize Tasks 3/4/5.
- `demand_baseline` has 10 feature columns; after lag-168 null-drop the effective training set is ~16,000 rows (train) / ~8,700 (val). Both should be >0.
- `mlflow.xgboost.autolog(log_models=False, ...)` (train.py:236-240) means no model artifact is saved. That is by design — stepped metrics on the parent run are what matters for the demo.

**VALIDATE**:
```bash
# Feature Parquet exists and has expected columns
uv run python -c "
import polars as pl
from windcast.features.registry import get_feature_set
df = pl.read_parquet('data/features/spain_demand_features.parquet')
fs = get_feature_set('demand_baseline')
missing = [c for c in fs.columns if c not in df.columns]
print('Rows:', len(df), 'Missing feature cols:', missing)
assert not missing
"

# MLflow parent run exists with h1_mae metric
uv run python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('enercast-spain_demand')
assert exp, 'experiment missing'
runs = client.search_runs([exp.experiment_id], filter_string=\"tags.enercast.feature_set = 'demand_baseline' and tags.enercast.run_type = 'parent'\")
assert runs, 'no parent run found for demand_baseline'
r = runs[0]
print('Parent run:', r.info.run_name)
print('Metrics h1_mae / h48_mae:', r.data.metrics.get('h1_mae'), '/', r.data.metrics.get('h48_mae'))
print('Skill h1 / h48:', r.data.metrics.get('h1_skill_score'), '/', r.data.metrics.get('h48_skill_score'))
assert 'h1_mae' in r.data.metrics
assert 'h48_mae' in r.data.metrics
"
```
**Expected**: Non-empty metrics, `h1_skill_score > 0` (should handily beat persistence at 1h), `h48_mae` present.

---

### Task 4: BUILD features + TRAIN `demand_enriched`

**IMPLEMENT**:
```bash
# Env vars from Task 3 must still be exported in this shell
uv run python scripts/build_features.py --domain demand --feature-set demand_enriched
uv run python scripts/train.py --domain demand --dataset spain_demand --feature-set demand_enriched
```

**PATTERN**: Same as Task 3.
**GOTCHA**: Adds `temperature_c`, `heating_degree_days`, `cooling_degree_days`, `load_mw_roll_mean_24`, `load_mw_roll_std_24`, `load_mw_roll_mean_168` — 6 new columns. HDD/CDD are computed in `_add_temperature_features` using 18°C / 24°C base; these can be zero in spring/fall, which is expected.
**VALIDATE**:
```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('enercast-spain_demand')
runs = client.search_runs([exp.experiment_id], filter_string=\"tags.enercast.feature_set = 'demand_enriched' and tags.enercast.run_type = 'parent'\")
assert runs
print('enriched h1_mae:', runs[0].data.metrics['h1_mae'], 'h48_mae:', runs[0].data.metrics['h48_mae'])
"
```

---

### Task 5: BUILD features + TRAIN `demand_full`

**IMPLEMENT**:
```bash
uv run python scripts/build_features.py --domain demand --feature-set demand_full
uv run python scripts/train.py --domain demand --dataset spain_demand --feature-set demand_full
```

**PATTERN**: Same as Tasks 3/4.
**GOTCHA**:
- `demand_full` adds `wind_speed_ms`, `humidity_pct`, `price_eur_mwh`, `price_lag1`, `price_lag24`, `is_holiday` — six passthrough/lag features. `wind_speed_ms` and `humidity_pct` come from the Spain weather CSV aggregate (columns already present in the processed Parquet — DO NOT pass `--weather-db`, it would try to look up a cached Open-Meteo Spain that does not exist).
- `is_holiday` is cast to `Int8` inside `build_demand_features` (demand.py:57) to be XGBoost-compatible.
- `price_eur_mwh` has some nulls in the Kaggle source — `drop_nulls()` at the end of feature building handles them. Row count may drop a few hundred vs. enriched.

**VALIDATE**:
```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('enercast-spain_demand')
runs = client.search_runs([exp.experiment_id], filter_string=\"tags.enercast.run_type = 'parent'\", order_by=['attributes.start_time ASC'])
assert len(runs) == 3, f'Expected 3 parent runs, got {len(runs)}'
print(f'{\"feature_set\":<18} {\"h1_mae\":>10} {\"h48_mae\":>10} {\"h1_skill\":>10} {\"h48_skill\":>10}')
for r in runs:
    fs = r.data.tags.get('enercast.feature_set', '?')
    m = r.data.metrics
    print(f'{fs:<18} {m.get(\"h1_mae\", float(\"nan\")):>10.1f} {m.get(\"h48_mae\", float(\"nan\")):>10.1f} {m.get(\"h1_skill_score\", float(\"nan\")):>10.3f} {m.get(\"h48_skill_score\", float(\"nan\")):>10.3f}')
"
```
**Expected**: 3 rows, `demand_full` should show the lowest MAE (addition of temperature + HDD/CDD is the biggest contributor historically). All `h1_skill_score` should be ≥ 0 (persistence is very strong at 1h for demand, so 0 is acceptable — the skill story for demand is usually at h12–h48).

---

### Task 6: Generate comparison artifacts

**IMPLEMENT**:
```bash
uv run python scripts/compare_runs.py --experiment enercast-spain_demand
```

**PATTERN**: Same invocation pattern as Pass 6b Kelmarsh comparison. Outputs MAE + skill bar charts to `reports/`.
**GOTCHA**: `compare_runs.py` reads parent-level `h{n}_mae` / `h{n}_skill_score` flat metrics — those are bubbled up from children by the re-collection block in `train.py:413-425`. They exist after the training loop; no extra step needed.
**VALIDATE**:
```bash
ls -la reports/comparison_enercast-spain_demand_*.png
```
**Expected**: Two PNGs (`mae` and `skill`) with non-zero size.

---

### Task 7: UPDATE `.claude/STATUS.md`

**IMPLEMENT**: Append the Pass 7 results to `.claude/STATUS.md`:
- Mark Pass 7 `[x]` in the task table
- Add a "Pass 7 — Demand end-to-end (Thu 9 April)" section mirroring Pass 6's format
- Paste the MAE + skill table captured from Task 5's validation query
- Add a one-line key result (e.g., "Same pipeline, zero core changes — Kelmarsh + Spain in one MLflow UI")

**PATTERN**: Mirror the "Pass 6b" block at `.claude/STATUS.md:116-139`.
**GOTCHA**: The numbers in the Pass 6 table are validation-set metrics — keep the same convention for Pass 7 (train.py logs val metrics by default). Remember to convert MAE from kW (wind) to MW (demand) — load is in MW, so all metrics are in MW units.
**VALIDATE**:
```bash
grep -n "Pass 7" .claude/STATUS.md
```

---

### Task 8: Regression check

**IMPLEMENT**: Re-run fast quality gates to ensure the ingest/train didn't drop any wind tests.
```bash
uv run ruff check src/ tests/ scripts/
uv run pytest tests/ -q
```
**Expected**: ruff clean; pytest `267 passed` (or whatever the current count is — should not decrease).

---

## TESTING STRATEGY

This pass is **executional, not additive**. No new unit tests should be created — the demand domain already has 50 passing tests. The "tests" for this pass are the validation queries embedded in each task.

### Validation Coverage

- **Data shape**: Row counts and date ranges checked after parse (Task 2).
- **Schema**: `validate_demand_schema` runs inside `ingest_spain_demand.py` (line 60-64); non-zero exit if broken.
- **Feature presence**: Every feature-set column checked present in the built Parquet (Tasks 3 validate block).
- **MLflow integrity**: Parent + child runs and key metrics queried via `MlflowClient.search_runs` after each training task.
- **Regression**: Full `pytest tests/` at the end (Task 8).

### Edge Cases to Watch

- **Only 4 years of data** — the whole pass hinges on the `WINDCAST_TRAIN_YEARS=2 WINDCAST_VAL_YEARS=1` override. If the shell session is reset between tasks, re-export.
- **`load_mw` nulls (36 rows)** — should survive parse, get dropped in feature build via `drop_nulls()`.
- **Spain weather has leading space in "Barcelona"** — handled by `.str.strip_chars()` in `spain_demand._read_weather_csv` (line 109). Do not re-introduce.
- **Valencia wind speed outliers > 50 m/s** — filtered to null in `_read_weather_csv` (line 131-136).
- **`is_holiday` XGBoost incompatibility** — the bool column is cast to `Int8` inside `build_demand_features` but only for `demand_full` (demand.py:57). `demand_baseline` and `demand_enriched` never touch `is_holiday` so it never enters the feature matrix.
- **lru_cache on `get_settings`** — do not call `get_settings()` interactively in the same Python process before exporting env vars; restart the shell or process between reconfigurations.

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and correct demand results.

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
```
**Expected**: All clean. No code is changing in this pass, so these should pass identically to the pre-pass baseline.

### Level 2: Unit Tests

```bash
uv run pytest tests/ -q
```
**Expected**: 267 passing (or higher — confirm against `git show HEAD:.claude/STATUS.md` baseline count).

### Level 3: Integration (Pipeline Run)

The 5 script executions in Tasks 2-6 ARE the integration test. Their validation blocks double as assertions.

### Level 4: Manual Validation — MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db &
# Then open http://localhost:5000
```
In the UI:
1. Confirm **two experiments**: `enercast-kelmarsh` and `enercast-spain_demand`
2. In `enercast-spain_demand`, confirm 3 parent runs (baseline/enriched/full), each with 5 children
3. Select all 3 parents → Compare → Charts tab → verify 3 lines on the `mae_by_horizon_min` and `skill_score_by_horizon_min` stepped charts
4. Parent run description Markdown should render the feature set name and val-set summary table

### Level 5: Cross-domain Comparison

```bash
uv run python scripts/compare_runs.py --experiment enercast-kelmarsh
uv run python scripts/compare_runs.py --experiment enercast-spain_demand
ls reports/
```
**Expected**: 4 PNGs total (MAE + skill for each experiment), all non-zero size.

---

## ACCEPTANCE CRITERIA

- [ ] `data/raw/spain/{energy_dataset.csv,weather_features.csv}` present, shapes match expected (35,064×29 and 178,396×17)
- [ ] `data/processed/spain_demand.parquet` exists, 11 columns matching `DEMAND_SCHEMA`, ≥30,000 rows, qc_ok > 95%
- [ ] MLflow experiment `enercast-spain_demand` exists with exactly 3 parent runs tagged `enercast.run_type='parent'`, one per feature set
- [ ] Each parent run has 5 child runs and flat metrics `h1_mae`, `h6_mae`, `h12_mae`, `h24_mae`, `h48_mae` + corresponding `h{n}_skill_score`
- [ ] Each parent run has stepped metrics `mae_by_horizon_min`, `skill_score_by_horizon_min` that render as a line chart in MLflow Charts tab
- [ ] At least one feature set produces `h6_skill_score ≥ 0.05` (demand has strong daily seasonality — beating persistence at 6h should be easy)
- [ ] `demand_full` should have lower MAE than `demand_baseline` on ≥3 of 5 horizons (temperature + HDD features should help)
- [ ] `reports/comparison_enercast-spain_demand_mae.png` and `...skill.png` generated and non-empty
- [ ] `.claude/STATUS.md` updated with Pass 7 section + real metrics table
- [ ] `uv run pytest tests/ -q` passes with no new failures
- [ ] `uv run ruff check src/ tests/ scripts/` clean

---

## COMPLETION CHECKLIST

- [ ] Task 1: Spain CSVs downloaded and shape-verified
- [ ] Task 2: Ingest produces Parquet with valid schema + sane QC summary
- [ ] Task 3: `demand_baseline` parent + 5 children in MLflow
- [ ] Task 4: `demand_enriched` parent + 5 children in MLflow
- [ ] Task 5: `demand_full` parent + 5 children in MLflow, cross-feature-set comparison table printed
- [ ] Task 6: Comparison PNGs in `reports/`
- [ ] Task 7: STATUS.md updated
- [ ] Task 8: ruff clean, pytest green
- [ ] Level 4 manual MLflow UI check passed (3 parents side-by-side in stepped line chart)
- [ ] Both `enercast-kelmarsh` and `enercast-spain_demand` visible in one MLflow UI — the demo-critical artifact

---

## NOTES

### Why no code changes

STATUS.md explicitly frames Pass 7 as "zero core changes" proof. The existing scripts already:
- Accept `--domain demand` (train.py:124, build_features.py:31, evaluate.py:160)
- Dispatch on domain for target/group/lag1 cols (train.py:28, evaluate.py:25)
- Use `DOMAIN_RESOLUTION[demand]=60` for stepped metric step-size (config.py:140)
- Handle hourly resolution in persistence baseline (train.py:354 via `lag1_col`)

If anything requires code changes during the run, first confirm it is a real bug, not a config issue. Document it inline in this plan under NOTES and fix minimally.

### Why skip evaluate.py

`scripts/train.py` sets `mlflow.xgboost.autolog(log_models=False, ...)` so no model artifact is saved to MLflow. `scripts/evaluate.py` relies on `mlflow.xgboost.load_model(...)` from the child runs (evaluate.py:126) — which will fail with "no model artifact" for both wind and demand. Pass 7 captures validation-set metrics from `train.py` directly, which are sufficient for the presentation.

If test-set metrics on demand become a hard requirement later, the fix is either to re-enable `log_models=True` OR re-run the test-set inference in-process at the end of `train.py` instead of a separate script. **Not in scope for Pass 7.**

### Why the env-var override and not a CLI flag

Adding `--train-years` / `--val-years` to `train.py` is a 3-line change that would be cleaner, but:
- It would violate the "zero core changes" framing of Pass 7
- The env-var mechanism already exists (`WINDCAST_` prefix, `pydantic-settings`) and is the documented way to override per-invocation
- It keeps Pass 7's diff minimal (only STATUS.md + data files)

If the pattern proves annoying, add CLI overrides in a post-Pass-7 cleanup task.

### Expected demand metrics (rough sanity band)

Based on typical Spain national load (20,000–40,000 MW range, strong daily/weekly seasonality, moderate weather dependence):

| Horizon | Expected MAE band | Expected skill vs. persistence |
|---------|-------------------|-------------------------------|
| h1 (1h) | 200–500 MW | 0.0–0.2 (persistence strong at 1h) |
| h6 (6h) | 600–1200 MW | 0.1–0.3 |
| h24 (24h) | 800–1600 MW | 0.3–0.5 (daily seasonality helps a lot) |
| h48 (48h) | 1000–2000 MW | 0.3–0.5 |

If skills are dramatically outside these bands (negative at h24 or > 0.7 anywhere), inspect the persistence computation — `lag1` at hourly resolution means "1h ago" which is a strong baseline for h1 but degrades for h>1.

### Confidence score: 8/10

High confidence because:
- All code exists and is unit-tested (50/50 demand tests passing)
- Multi-domain CLI dispatch has been exercised in code review
- Pass 6 proved the same MLflow tooling on Kelmarsh end-to-end

Minor risks (-2):
- Kaggle download step may need manual intervention (no Kaggle CLI installed, credentials may need setup)
- Real data always reveals surprises (outlier rows, schema drift at CSV edges)
- `_temporal_split` with `start.replace(year=...)` can hit Feb-29 edge cases (Spain data starts 2015-01-01, not a leap-year concern here, but worth flagging)
