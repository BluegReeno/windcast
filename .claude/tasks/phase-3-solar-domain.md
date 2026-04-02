# Feature: Phase 3 — Solar Domain

## Goal
Add solar PV forecasting as a third domain to complete the 3-domain trifecta (wind, demand, solar) with zero core pipeline changes.

## Context
- **PRD Reference**: `.claude/PRD.md` — Phase 3, Solar Domain
- **Plan**: `.claude/plans/phase-3-solar-domain.md`

## Tasks

### Phase 1: Foundation — Schema + Config
- [x] Task 1: Create `src/windcast/data/solar_schema.py` ✓ 2026-04-02
- [x] Task 2: Create `tests/data/test_solar_schema.py` ✓ 2026-04-02
- [x] Task 3: Update `src/windcast/config.py` with solar config ✓ 2026-04-02
- [x] Task 4: Update `tests/test_config.py` with solar tests ✓ 2026-04-02

### Phase 2: Parser — PVDAQ
- [x] Task 5: Create `src/windcast/data/pvdaq.py` ✓ 2026-04-02
- [x] Task 6: Create `tests/data/test_pvdaq.py` ✓ 2026-04-02

### Phase 3: Solar QC
- [x] Task 7: Create `src/windcast/data/solar_qc.py` ✓ 2026-04-02
- [x] Task 8: Create `tests/data/test_solar_qc.py` ✓ 2026-04-02

### Phase 4: Features + Registry
- [x] Task 9: Update `src/windcast/data/__init__.py` ✓ 2026-04-02
- [x] Task 10: Create `src/windcast/features/solar.py` ✓ 2026-04-02
- [x] Task 11: Update `src/windcast/features/registry.py` with solar sets ✓ 2026-04-02
- [x] Task 12: Update `src/windcast/features/__init__.py` ✓ 2026-04-02
- [x] Task 13: Create `tests/features/test_solar.py` ✓ 2026-04-02
- [x] Task 14: Update `tests/features/test_registry.py` ✓ 2026-04-02

### Phase 5: Script Integration
- [x] Task 15: Create `scripts/ingest_pvdaq.py` ✓ 2026-04-02
- [x] Task 16: Update `scripts/build_features.py` with --domain solar ✓ 2026-04-02
- [x] Task 17: Update `scripts/train.py` with --domain solar ✓ 2026-04-02
- [x] Task 18: Update `scripts/evaluate.py` with --domain solar ✓ 2026-04-02

### Phase 6: Validation
- [x] Task 19: Run full validation suite (ruff, pyright, pytest) ✓ 2026-04-02

## Completion
- **Started**: 2026-04-02
- **Completed**: 2026-04-02
- **Commit**: —
