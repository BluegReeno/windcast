# Feature: Pass 11 — Model Registry Integration

## Goal
Make trained EnerCast models servable by logging them with MLflow signatures and registering them to the Model Registry.

## Context
- **PRD Reference**: `.claude/PRD.md` — Path to Production (Passes 11-13)
- **Plan**: `.claude/plans/pass11-model-registry.md`
- **Related Files**: `training/harness.py`, `training/backends.py`, `scripts/train.py`

## Tasks

### Phase 1: Foundation
- [x] Create `src/windcast/models/autogluon_pyfunc.py` — custom PythonModel wrapper for AutoGluon ✓ 2026-04-10
- [x] Add `log_model()` to `TrainingBackend` Protocol in `harness.py` ✓ 2026-04-10

### Phase 2: Backend Implementations
- [x] Implement `XGBoostBackend.log_model()` in `backends.py` ✓ 2026-04-10
- [x] Implement `AutoGluonBackend.log_model()` in `backends.py` ✓ 2026-04-10

### Phase 3: Integration
- [x] Wire `log_model()` into `run_training()` + add registration logic ✓ 2026-04-10
- [x] Add `--no-log-models`, `--register`, `--model-name` CLI flags to `train.py` ✓ 2026-04-10

### Phase 4: Testing
- [x] Update `MockBackend` in `test_harness.py` + pass `log_models=False` ✓ 2026-04-10
- [x] Create `tests/training/test_model_registry.py` — integration tests ✓ 2026-04-10

### Validation
- [x] ruff check + format + pyright pass ✓ 2026-04-10
- [x] pytest tests/training/ pass (15/15) ✓ 2026-04-10
- [x] Full test suite passes (325/325) ✓ 2026-04-10

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/models/autogluon_pyfunc.py` | Create | Custom MLflow PythonModel for AutoGluon |
| `src/windcast/training/harness.py` | Modify | Add `log_model()` to Protocol + wire into `run_training()` |
| `src/windcast/training/backends.py` | Modify | Implement `log_model()` in XGBoost + AutoGluon backends |
| `scripts/train.py` | Modify | Add `--no-log-models`, `--register`, `--model-name` flags |
| `tests/training/test_harness.py` | Modify | Update MockBackend + existing test |
| `tests/training/test_model_registry.py` | Create | Integration tests for model logging |

## Notes
- Keep `log_models=False` in XGBoost autolog to avoid double-logging
- `infer_signature()` requires pandas, not Polars
- MLflow 3.x: use `name=` not `artifact_path=` (deprecated)
- MLflow 3.x: `log_model()` returns `ModelInfo` with `.model_uri` — models stored in `mlruns/<exp>/models/<model_id>/` NOT in `<run_id>/artifacts/`
- AutoGluon has no built-in MLflow flavor — needs custom pyfunc wrapper
- Registration uses `model_uri` from `log_model()` return value, not `runs:/<run_id>/<name>` (deprecated)

## Completion
- **Started**: 2026-04-10
- **Completed**: 2026-04-10
- **Commit**: (fill after commit)
