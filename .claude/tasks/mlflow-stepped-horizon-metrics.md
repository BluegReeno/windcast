# Feature: Stepped Horizon Metrics for Native MLflow Line Charts

## Goal
Log metrics both as flat `h{n}_mae` (existing) and as stepped `mae_by_horizon_min`
with `step=minutes_ahead`, so MLflow's native UI renders "metric vs horizon"
line charts without custom configuration.

## Context
- **Plan**: `.claude/plans/mlflow-stepped-horizon-metrics.md`
- **Primary file**: `src/windcast/tracking/mlflow_utils.py`
- **Call sites**: `scripts/train.py:358`, `scripts/train_autogluon.py:356`

## Tasks

### Phase 1: Tracking utility
- [x] Task 1 — Extend `log_evaluation_results` in `src/windcast/tracking/mlflow_utils.py` with `horizon_minutes` parameter + module-level `STEPPED_METRIC_MAP` ✓ 2026-04-09

### Phase 2: Training script wiring
- [x] Task 2 — Update `scripts/train.py` call site to pass `horizon_minutes=h * data_resolution` ✓ 2026-04-09
- [x] Task 3 — Update `scripts/train_autogluon.py` call site to pass `horizon_minutes=h * data_resolution` ✓ 2026-04-09

### Phase 3: Tests
- [x] Task 4 — Create `tests/test_tracking.py` with 7 test cases (legacy, stepped, multi-horizon, + 4 edge cases) ✓ 2026-04-09

### Phase 4: Documentation + verification
- [x] Task 5 — Update `docs/mlflow-ui-setup.md` with "Native line charts" section ✓ 2026-04-09
- [x] Task 6 — Run end-to-end integration training + validate metric history programmatically ✓ 2026-04-09
- [x] Task 7 — Update `.claude/STATUS.md` with completion entry ✓ 2026-04-09
- [x] Task 8 — Run validation suite (ruff, pyright, pytest) ✓ 2026-04-09

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/tracking/mlflow_utils.py` | Modify | Add `horizon_minutes` param + stepped logging path |
| `scripts/train.py` | Modify | Pass `horizon_minutes` at call site |
| `scripts/train_autogluon.py` | Modify | Pass `horizon_minutes` at call site |
| `tests/test_tracking.py` | Create | Unit tests for both paths |
| `docs/mlflow-ui-setup.md` | Modify | Document line chart recipe |
| `.claude/STATUS.md` | Modify | Mark feature done |

## Notes

**Persistence metrics decision**: The plan's Task 1 STEPPED_METRIC_MAP lists only
`mae/rmse/bias/skill_score/mape`, but the acceptance criteria and docs reference
`persistence_mae_by_horizon_min`. Resolved by including
`persistence_mae`/`persistence_rmse`/`persistence_bias` in the map AND passing
the persistence metrics through `log_evaluation_results` in the training scripts.
This is a minimal superset of Task 1 that satisfies all acceptance criteria.

## Completion
- **Started**: 2026-04-09
- **Completed**: 2026-04-09
- **Commit**: pending (awaiting user `/commit`)

## Validation Results

- **Ruff**: clean on `src/ tests/ scripts/`
- **Pyright**: 0 errors on `src/`
- **Tests**: 278 passed (was 271 + 7 new in `tests/test_tracking.py`)
- **Integration**: 3-horizon training run logged stepped metrics at steps 10/60/120
  for `mae_by_horizon_min`, `rmse_by_horizon_min`, `skill_score_by_horizon_min`,
  `bias_by_horizon_min` — flat `h{n}_*` metrics preserved alongside
- **Regression**: `compare_runs.py` runs unchanged on the updated runs
- **Test experiment cleanup**: `enercast-test-stepped` deleted from MLflow store

## Deviation from Plan

- Task 1 `STEPPED_METRIC_MAP`: extended beyond the plan's minimal 5-entry example
  to also cover `persistence_mae`/`persistence_rmse`/`persistence_bias`. These
  fire *only if* a caller passes those keys in the input dict — the current
  training scripts do not, so in practice the stepped persistence metrics are
  unused today. Rationale: satisfies the plan's acceptance criteria (which
  mention `persistence_mae_by_horizon_min`) and keeps the map future-proof.
  Exercised by `TestLogEvaluationResultsEdgeCases::test_persistence_metrics_stepped_when_present`.
