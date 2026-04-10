# Feature: Training Harness + MLflow Lineage Tags

## Goal
Unify duplicated training scripts into a single harness with a Backend Protocol, and auto-log MLflow lineage tags on every run.

## Context
- **Plan**: `.claude/plans/training-harness-mlflow-lineage.md`
- **Key files**: `scripts/train.py`, `src/windcast/training/`

## Tasks

### Phase 1: Training Package Foundation
- [x] Task 1: Create `src/windcast/training/__init__.py` — exports ✓ 2026-04-10
- [x] Task 2: Create `src/windcast/training/harness.py` — Protocol + shared utilities ✓ 2026-04-10
- [x] Task 3: Create `src/windcast/training/lineage.py` — git info + lineage tags ✓ 2026-04-10

### Phase 2: Backend Implementations
- [x] Task 4: Create `src/windcast/training/backends.py` — XGBoost + AutoGluon backends ✓ 2026-04-10

### Phase 3: Training Harness
- [x] Task 5: Implement `run_training()` in harness.py — the main training loop ✓ 2026-04-10

### Phase 4: Unified CLI + Cleanup
- [x] Task 6: Rewrite `scripts/train.py` as thin CLI ✓ 2026-04-10
- [x] Task 7: Delete `scripts/train_autogluon.py` — pending user confirmation

### Phase 5: Lineage Integration
- [x] Task 8: Update `scripts/log_tso_baseline.py` — add lineage tags ✓ 2026-04-10
- [x] Task 9: Update `scripts/compare_runs.py` — data_quality filter ✓ 2026-04-10

### Phase 6: Backfill + Exports
- [x] Task 10: Create `scripts/annotate_mlflow_runs.py` — backfill script ✓ 2026-04-10
- [x] Task 11: Update `src/windcast/training/__init__.py` — final exports ✓ 2026-04-10

### Phase 7: Tests
- [x] Task 12: Create `tests/training/test_harness.py` — 11 tests ✓ 2026-04-10
- [x] Task 13: Create `tests/training/test_lineage.py` — 3 tests ✓ 2026-04-10

### Phase 8: Docs + Validation
- [x] Task 14: Update docs references (train_autogluon → train.py --backend autogluon) ✓ 2026-04-10
- [x] Validation: ruff + pyright + pytest pass (321 tests, 0 failures) ✓ 2026-04-10

## Completion
- **Started**: 2026-04-10
- **Completed**: 2026-04-10
- **Commit**: (pending)
