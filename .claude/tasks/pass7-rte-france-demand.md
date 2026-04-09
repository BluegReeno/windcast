# Feature: Pass 7 — RTE France Demand Domain (éCO2mix local files)

## Goal
Replace the Spain Kaggle dataset with 11 years of real French national load from locally-downloaded RTE éCO2mix annual definitive files, and refactor demand feature sets to use forward-looking NWP (rearview → windshield narrative). Ship with an RTE TSO day-ahead benchmark as the killer comparison slide.

## Context
- **Plan**: `.claude/plans/pass7-rte-france-demand.md`
- **Data**: `data/ecomix-rte/eCO2mix_RTE_Annuel-Definitif_<YYYY>.zip` (2014-2024)
- **Preceding pass**: Pass 6b (stepped horizon metrics + SQLite MLflow backend)

## Tasks

### Phase 1: Schema + Config Foundation
- [x] Task 1: Extend `DEMAND_SCHEMA` with `tso_forecast_mw` (Float64, nullable) ✓ 2026-04-09
- [x] Task 2: Update `tests/data/test_demand_schema.py` column counts 11→12 ✓ 2026-04-09
- [x] Task 3: Update `tests/data/test_spain_demand.py` column counts 11→12 ✓ 2026-04-09 (no changes needed; tests green)
- [x] Task 4: Add `RTE_FRANCE` DemandDatasetConfig in `config.py` ✓ 2026-04-09
- [x] Task 5: Add `RTE_FRANCE_WEATHER` in `weather/registry.py` ✓ 2026-04-09

### Phase 2: RTE éCO2mix Parser
- [x] Task 6: Create `src/windcast/data/rte_france.py` (~150 lines) ✓ 2026-04-09
- [x] Task 7: Smoke-test parser against real 2023 + 2014 files ✓ 2026-04-09 (all 11 years parse, 96,421 hourly rows, load 29-96 GW)
- [x] Task 8: Create `tests/data/test_rte_france.py` with in-memory fixtures ✓ 2026-04-09 (8 tests, all passing)

### Phase 3: QC + Feature Set Refactor
- [x] Task 9: Add `FRANCE_HOLIDAYS` + `HOLIDAYS_BY_DATASET` dispatch in `demand_qc.py` ✓ 2026-04-09
- [x] Task 10: Add French holiday test cases in `tests/data/test_demand_qc.py` ✓ 2026-04-09
- [x] Task 11: Refactor `DEMAND_ENRICHED` / `DEMAND_FULL` in `features/registry.py` ✓ 2026-04-09
- [x] Task 12: Update `build_demand_features` for NWP-aware HDD/CDD ✓ 2026-04-09
- [x] Task 13: Update `tests/features/test_demand.py` ✓ 2026-04-09

### Phase 4: CLI + Backfill
- [x] Task 14: Add `--dataset` to `scripts/build_features.py` + multi-point NWP dispatch ✓ 2026-04-09
- [x] Task 15: Wire `--dataset` into parquet path in `scripts/train.py` ✓ 2026-04-09
- [x] Task 16: Backfill weighted 8-city France NWP into `data/weather.db` ✓ 2026-04-09 (96,432 hourly rows; wattcast TEMP_POINTS strategy)
- [x] Task 17: Create `scripts/ingest_rte_france.py` ✓ 2026-04-09
- [x] Task 18: Raise `DemandQCConfig.max_load_mw` to 100,000 ✓ 2026-04-09

**Note (deviation from plan)**: At user's request, switched from single-point Paris to
8-city demand-weighted national average via new `WeightedWeatherConfig` (Paris 0.30,
Lyon 0.15, Tours 0.14, Lille 0.10, Bordeaux 0.08, Toulouse 0.08, Strasbourg 0.08,
Marseille 0.07). This mirrors the wattcast `TEMP_POINTS` strategy and is a better
proxy for continental French demand than Paris alone.

### Phase 5: Run Pipeline + Benchmarks
- [x] Task 19: Run real ingestion → `data/processed/rte_france.parquet` ✓ 2026-04-09 (96,421 rows, 100% QC_OK, 2,904 holiday hours)
- [x] Task 20: Run 3 training feature sets (baseline/enriched/full) ✓ 2026-04-09
- [x] Task 21: Create + run `scripts/log_tso_baseline.py` ✓ 2026-04-09 (MAE 1,205 MW)
- [x] Task 22: Compare runs + export PNGs via `scripts/compare_runs.py` ✓ 2026-04-09 (4 parent runs, 2 PNGs)
- [x] Task 23: Update `.claude/STATUS.md` with Pass 7 results ✓ 2026-04-09
- [x] Task 24: Regression check (ruff, pyright, pytest) ✓ 2026-04-09 (294 tests, 0 pyright errors, ruff clean)

## Completion
- **Started**: 2026-04-09
- **Completed**: 2026-04-09
- **Commit**: (pending — run /commit to create)

## Results summary

**Val set 2022-2023 (17,518 hourly rows, 8-city weighted NWP):**

| Horizon | Baseline MAE | Enriched MAE | Full MAE |
|---------|-------------|--------------|----------|
| h1  | 839 MW | 782 MW | **766 MW** |
| h6  | 1,430 MW | 1,168 MW | **1,130 MW** |
| h12 | 1,634 MW | 1,377 MW | **1,254 MW** |
| h24 | 1,506 MW | 1,485 MW | **1,223 MW** |
| h48 | 2,121 MW | 2,114 MW | **1,643 MW** |

**RTE TSO D-1 benchmark at h24: 1,205 MW MAE**
**demand_full at h24: 1,223 MW MAE** — within **1.5%** of RTE's own forecast.

## Notes
- File format is TSV Latin-1 despite `.xls` extension — no openpyxl/xlrd needed
- Date format is ISO `YYYY-MM-DD` (PDF spec is wrong on this point)
- Load at native 30-min, TSO forecast at native 15-min → aggregated to hourly mean
- Spain parser retained as 2nd reference implementation, not deleted

## Completion
- **Started**: 2026-04-09
- **Completed**: (fill when done)
- **Commit**: (fill when done)
