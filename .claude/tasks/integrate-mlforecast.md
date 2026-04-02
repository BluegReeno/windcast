# Feature: Integrate mlforecast (Nixtla) as ML Training Backend

## Goal
Add mlforecast as an alternative ML training backend with automatic lag management and multi-horizon strategies.

## Context
- **Plan Reference**: `.claude/plans/integrate-mlforecast.md`
- **Related Files**: models/, features/, scripts/

## Tasks

### Phase 1: Foundation
- [x] Add mlforecast dependency to pyproject.toml + uv sync ✓ 2026-04-02

### Phase 2: Exogenous Features
- [x] Create src/windcast/features/exogenous.py (domain-specific features, no lags/rolling) ✓ 2026-04-02
- [x] Add exogenous feature set definitions to registry.py (9 new entries) ✓ 2026-04-02
- [x] Update src/windcast/features/__init__.py with new exports ✓ 2026-04-02

### Phase 3: MLForecast Model Wrapper
- [x] Create src/windcast/models/mlforecast_model.py ✓ 2026-04-02
- [x] Update src/windcast/models/__init__.py with new exports ✓ 2026-04-02

### Phase 4: Training Script
- [x] Create scripts/train_mlforecast.py ✓ 2026-04-02

### Phase 5: Testing
- [x] Create tests/models/test_mlforecast_model.py ✓ 2026-04-02
- [x] Create tests/features/test_exogenous.py ✓ 2026-04-02

### Phase 6: Validation
- [x] Run full validation suite (ruff, pyright, pytest) ✓ 2026-04-02

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `pyproject.toml` | Modify | Add mlforecast dependency |
| `src/windcast/features/exogenous.py` | Create | Domain-specific exogenous feature builders |
| `src/windcast/features/registry.py` | Modify | Add 9 exogenous feature set definitions |
| `src/windcast/features/__init__.py` | Modify | Export new builders |
| `src/windcast/models/mlforecast_model.py` | Create | MLForecast wrapper |
| `src/windcast/models/__init__.py` | Modify | Export new symbols |
| `scripts/train_mlforecast.py` | Create | Training CLI script |
| `tests/models/test_mlforecast_model.py` | Create | Unit tests for wrapper |
| `tests/features/test_exogenous.py` | Create | Unit tests for exogenous builders |

## Completion
- **Started**: 2026-04-02
- **Completed**: 2026-04-02
- **Commit**: (pending /commit)
