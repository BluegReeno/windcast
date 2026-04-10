# Feature: Per-Dataset Split Configuration + CLI Override

## Goal
Move train_years/val_years from global settings to per-dataset configs, add CLI overrides, consolidate duplicated temporal_split().

## Context
- **Plan**: `.claude/plans/per-dataset-split-config.md`
- **Related Files**: config.py, harness.py, train.py, evaluate.py, log_tso_baseline.py, train_mlforecast.py, test_harness.py

## Tasks

### Phase 1: Config
- [x] Add train_years/val_years to DatasetConfig, DemandDatasetConfig, SolarDatasetConfig ✓ 2026-04-10

### Phase 2: Harness
- [x] Add timestamp_col param to temporal_split(), add train_years/val_years to run_training(), log to MLflow ✓ 2026-04-10

### Phase 3: CLI args
- [x] Add --train-years/--val-years to train.py with resolution chain ✓ 2026-04-10
- [x] Replace duplicate _temporal_split in evaluate.py, add CLI args ✓ 2026-04-10
- [x] Replace inline split in log_tso_baseline.py, add CLI args ✓ 2026-04-10
- [x] Replace duplicate _temporal_split in train_mlforecast.py, add CLI args ✓ 2026-04-10

### Phase 4: Tests + Validation
- [x] Update tests for new params ✓ 2026-04-10
- [x] Full validation (ruff, pyright, pytest) ✓ 2026-04-10

## Completion
- **Started**: 2026-04-10
- **Completed**: 2026-04-10
- **Commit**: (pending /commit)
