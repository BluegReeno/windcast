# Feature: Phase 2 — Demand Domain

## Goal
Add power demand forecasting as a second domain to prove pipeline genericity across wind and demand.

## Context
- **PRD Reference**: `.claude/PRD.md` — multi-domain standardization
- **Plan**: `.claude/plans/phase-2-demand-domain.md`

## Tasks

### Phase 1: Foundation — Schema + Config
- [x] Task 1: Create `src/windcast/data/demand_schema.py` ✓ 2026-04-02
- [x] Task 2: Create `tests/data/test_demand_schema.py` ✓ 2026-04-02
- [x] Task 3: Update `src/windcast/config.py` with demand config ✓ 2026-04-02
- [x] Task 4: Update `tests/test_config.py` with demand tests ✓ 2026-04-02

### Phase 2: Parser — Spain Demand
- [x] Task 5: Create `src/windcast/data/spain_demand.py` ✓ 2026-04-02
- [x] Task 6: Create `tests/data/test_spain_demand.py` ✓ 2026-04-02

### Phase 3: Demand QC
- [x] Task 7: Create `src/windcast/data/demand_qc.py` ✓ 2026-04-02
- [x] Task 8: Create `tests/data/test_demand_qc.py` ✓ 2026-04-02

### Phase 4: Exports + Features
- [x] Task 9: Update `src/windcast/data/__init__.py` ✓ 2026-04-02
- [x] Task 10: Add demand feature sets to `src/windcast/features/registry.py` ✓ 2026-04-02
- [x] Task 11: Create `src/windcast/features/demand.py` ✓ 2026-04-02
- [x] Task 12: Update `src/windcast/features/__init__.py` ✓ 2026-04-02
- [x] Task 13: Create `tests/features/test_demand.py` ✓ 2026-04-02
- [x] Task 14: Update `tests/features/test_registry.py` ✓ 2026-04-02

### Phase 5: Script Integration
- [x] Task 15: Create `scripts/ingest_spain_demand.py` ✓ 2026-04-02
- [x] Task 16: Update `scripts/build_features.py` with --domain flag ✓ 2026-04-02
- [x] Task 17: Update `scripts/train.py` with --domain flag ✓ 2026-04-02
- [x] Task 18: Update `scripts/evaluate.py` with --domain flag ✓ 2026-04-02

### Phase 6: Validation
- [x] Task 19: Run full validation suite (ruff, pyright, pytest) ✓ 2026-04-02

## Completion
- **Started**: 2026-04-02
- **Completed**: 2026-04-02
- **Commit**: (pending)
