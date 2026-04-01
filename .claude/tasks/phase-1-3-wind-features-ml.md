# Feature: Phase 1.3 — Wind Feature Engineering + ML Pipeline

## Goal
Build complete wind ML pipeline: feature engineering, persistence baseline, XGBoost training with MLflow, evaluation with skill scores.

## Context
- **PRD Reference**: `.claude/PRD.md` Phase 1.3
- **Plan**: `.claude/plans/phase-1-3-wind-features-ml.md`
- **Related Files**: `src/windcast/config.py`, `src/windcast/data/schema.py`, `src/windcast/data/qc.py`

## Tasks

### Phase 1: Feature Engineering Foundation
- [x] Task 1: Create `src/windcast/features/registry.py` — Feature set registry ✓ 2026-04-01
- [x] Task 2: Create `src/windcast/features/wind.py` — Wind feature builders ✓ 2026-04-01
- [x] Task 3: Update `src/windcast/features/__init__.py` — Exports ✓ 2026-04-01

### Phase 2: Models & Evaluation
- [x] Task 4: Create `src/windcast/models/persistence.py` — Persistence baseline ✓ 2026-04-01
- [x] Task 5: Create `src/windcast/models/evaluation.py` — Metrics & evaluation ✓ 2026-04-01
- [x] Task 6: Create `src/windcast/models/xgboost_model.py` — XGBoost wrapper ✓ 2026-04-01
- [x] Task 8: Update `src/windcast/models/__init__.py` — Exports ✓ 2026-04-01

### Phase 3: MLflow Tracking
- [x] Task 7: Create `src/windcast/tracking/` — MLflow utilities ✓ 2026-04-01

### Phase 4: CLI Scripts
- [x] Task 9: Create `scripts/build_features.py` — Feature building CLI ✓ 2026-04-01
- [x] Task 10: Create `scripts/train.py` — Training CLI ✓ 2026-04-01
- [x] Task 11: Create `scripts/evaluate.py` — Evaluation CLI ✓ 2026-04-01

### Phase 5: Testing & Validation
- [x] Task 12: Create tests for features, models, evaluation ✓ 2026-04-01
- [x] Task 13: Final validation — lint, format, type check, all tests pass ✓ 2026-04-01

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/features/registry.py` | Create | Feature set definitions |
| `src/windcast/features/wind.py` | Create | Wind feature engineering |
| `src/windcast/features/__init__.py` | Modify | Public exports |
| `src/windcast/models/persistence.py` | Create | Persistence baseline |
| `src/windcast/models/evaluation.py` | Create | Metrics and evaluation |
| `src/windcast/models/xgboost_model.py` | Create | XGBoost training wrapper |
| `src/windcast/models/__init__.py` | Modify | Public exports |
| `src/windcast/tracking/__init__.py` | Create | Package init |
| `src/windcast/tracking/mlflow_utils.py` | Create | MLflow helpers |
| `scripts/build_features.py` | Create | Feature building CLI |
| `scripts/train.py` | Create | Training CLI |
| `scripts/evaluate.py` | Create | Evaluation CLI |
| `tests/features/` | Create | Feature tests |
| `tests/models/` | Create | Model tests |

## Notes
- XGBoost 3.x moved `early_stopping_rounds` to constructor (no longer a `fit()` param)
- `mlflow.xgboost` needs pyright ignore for `reportPrivateImportUsage`

## Completion
- **Started**: 2026-04-01
- **Completed**: 2026-04-01
- **Commit**: (pending commit)
