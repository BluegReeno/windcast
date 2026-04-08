# Feature: Ingest Kelmarsh Real Data

## Goal
Fix the Kelmarsh parser to handle real Greenbyte CSV format (comment lines, case mismatch, ragged lines) and run ingestion on the 3.8 GB real dataset.

## Context
- **Plan**: `.claude/plans/ingest-kelmarsh-real-data.md`
- **Primary File**: `src/windcast/data/kelmarsh.py`
- **Tests**: `tests/data/test_kelmarsh.py`

## Tasks

### Phase 1: Fix Parser
- [x] Task 1: Fix signal map case — `"Rotor speed (rpm)"` → `"Rotor speed (RPM)"` ✓ 2026-04-08
- [x] Task 2: Add `_strip_comment_lines()` + ragged lines resilience ✓ 2026-04-08
- [x] Task 3: Use `Data Availability` column for QC flagging ✓ 2026-04-08
- [x] Task 4: Deduplicate timestamps in `parse_kelmarsh()` after concat ✓ 2026-04-08

### Phase 2: Update Tests
- [x] Task 5: Add real-format test with comment lines + fix mock header case ✓ 2026-04-08

### Phase 3: Run & Validate
- [x] Task 6: Run full ingestion on real data ✓ 2026-04-08
- [x] Task 7: Run full validation suite (ruff, pyright, pytest) ✓ 2026-04-08

## Files Modified

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/data/kelmarsh.py` | Modified | Fix parser for real CSV format |
| `tests/data/test_kelmarsh.py` | Modified | Add real-format tests |

## Notes
- Extra fix needed: numeric columns cast to Float64 after read, because `ignore_errors=True` infers some as str
- 2024 year CSVs are much larger (~300 cols) — takes ~15s per turbine vs ~0.5s for earlier years
- QC stats: 67.2% OK, 21.0% suspect, 11.8% bad — reasonable for 9 years of real SCADA data

## Completion
- **Started**: 2026-04-08
- **Completed**: 2026-04-08
- **Commit**: (pending /commit)
