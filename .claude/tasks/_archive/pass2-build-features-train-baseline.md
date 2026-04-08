# Feature: Pass 2 — Build Features + Train Baseline on Kelmarsh Real Data

## Goal
Run the full feature-engineering + training pipeline on real Kelmarsh data, get actual metrics in MLflow.

## Context
- **Plan**: `.claude/plans/pass2-build-features-train-baseline.md`
- **Related Files**: `scripts/build_features.py`, `scripts/train.py`, `src/windcast/features/wind.py`

## Tasks

### Phase 1: Build Features
- [x] Run `build_features.py --feature-set wind_baseline` on all 6 turbines ✓ 2026-04-08
- [x] Validate: 6 Parquets in `data/features/`, 272k-318k rows, 29 columns ✓ 2026-04-08

### Phase 2: Train Baseline
- [x] Run `train.py --turbine-id kwf1 --feature-set wind_baseline` ✓ 2026-04-08
- [x] Validate: MLflow run with 5 horizon runs, metrics logged ✓ 2026-04-08

### Phase 3: Verify Results
- [x] Check skill_score > 0 at h1 (model beats persistence) — skill=0.203 ✓ 2026-04-08
- [x] Note actual metrics for presentation use ✓ 2026-04-08

### Phase 4: Finalize
- [x] Update STATUS.md with Pass 2 results ✓ 2026-04-08

## Notes
- **Bug fixed**: `scripts/train.py` line 44 used `offset_by()` (Polars method) on a Python `datetime` object. Fixed to use `datetime.replace(year=...)`.
- **Metrics** (kwf1, wind_baseline): h1 MAE=120kW skill=0.203 | h6 MAE=210kW skill=0.097 | h12 MAE=259kW skill=0.091 | h24 MAE=334kW skill=0.107 | h48 MAE=432kW skill=0.130
- All skill scores positive — model beats persistence at every horizon, even h48 without NWP features

## Completion
- **Started**: 2026-04-08
- **Completed**: 2026-04-08
- **Commit**: (link to commit when done)
