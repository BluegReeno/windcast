# Feature: AutoGluon-Tabular as 3rd ML Backend

## Goal
Add AutoGluon-Tabular as a 3rd ML backend alongside XGBoost and mlforecast, proving framework pluggability with WN's own stack.

## Context
- **PRD Reference**: `.claude/PRD.md` — backend-agnostic architecture
- **Plan**: `.claude/plans/pass6-autogluon-tabular.md`

## Tasks

### Setup
- [x] Add `autogluon.tabular[all]>=1.2` to pyproject.toml ✓ 2026-04-08
- [x] `uv sync` completes without errors ✓ 2026-04-08

### Implementation
- [x] Create `src/windcast/models/autogluon_model.py` — AutoGluonConfig + train_autogluon ✓ 2026-04-08
- [x] Create `scripts/train_autogluon.py` — training script with MLflow logging ✓ 2026-04-08
- [x] Update `src/windcast/models/__init__.py` — add exports ✓ 2026-04-08

### Testing
- [x] Create `tests/models/test_autogluon_model.py` — config + integration tests ✓ 2026-04-08
- [x] Register `slow` pytest marker in pyproject.toml ✓ 2026-04-08

### Validation
- [x] `uv run ruff check src/ tests/ scripts/` — 0 errors ✓ 2026-04-08
- [x] `uv run ruff format --check src/ tests/ scripts/` — all formatted ✓ 2026-04-08
- [x] `uv run pyright src/` — 0 errors ✓ 2026-04-08
- [x] `uv run pytest tests/ -v` — 271 tests pass (267 + 4 new) ✓ 2026-04-08

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `pyproject.toml` | Modified | Added autogluon dep + slow marker |
| `src/windcast/models/autogluon_model.py` | Created | AutoGluonConfig + train_autogluon wrapper |
| `scripts/train_autogluon.py` | Created | Training script (same interface as train.py) |
| `tests/models/test_autogluon_model.py` | Created | 4 tests (config + integration) |
| `src/windcast/models/__init__.py` | Modified | Added AutoGluon exports |

## Completion
- **Started**: 2026-04-08
- **Completed**: 2026-04-08
