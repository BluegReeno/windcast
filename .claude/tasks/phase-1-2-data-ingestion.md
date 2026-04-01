# Feature: Phase 1.2 — Data Ingestion & QC Pipeline

## Goal
Implement the complete data ingestion pipeline: SCADA schema, Kelmarsh parser, QC pipeline, Open-Meteo client, and CLI script.

## Context
- **PRD Reference**: `.claude/PRD.md` sections 7.1–7.3
- **Plan**: `.claude/plans/phase-1-2-data-ingestion.md`
- **Related Files**: `src/windcast/data/`, `src/windcast/config.py`, `scripts/`, `tests/`

## Tasks

### Phase 1: Foundation
- [x] Task 1: Create `src/windcast/config.py` — Pydantic Settings ✓ 2026-04-01
- [x] Task 2: Create `src/windcast/data/schema.py` — Canonical SCADA schema ✓ 2026-04-01

### Phase 2: Core Parser
- [x] Task 3: Create `src/windcast/data/kelmarsh.py` — Kelmarsh v4 parser ✓ 2026-04-01

### Phase 3: Quality Control
- [x] Task 4: Create `src/windcast/data/qc.py` — QC pipeline ✓ 2026-04-01

### Phase 4: NWP Client
- [x] Task 5: Create `src/windcast/data/open_meteo.py` — Open-Meteo client ✓ 2026-04-01

### Phase 5: Wiring
- [x] Task 6: Update `src/windcast/data/__init__.py` — exports ✓ 2026-04-01
- [x] Task 7: Create `scripts/ingest_kelmarsh.py` — CLI script ✓ 2026-04-01

### Phase 6: Testing
- [x] Task 8: Create `tests/data/__init__.py` ✓ 2026-04-01
- [x] Task 9: Create `tests/test_config.py` ✓ 2026-04-01
- [x] Task 10: Create `tests/data/test_schema.py` ✓ 2026-04-01
- [x] Task 11: Create `tests/data/test_kelmarsh.py` ✓ 2026-04-01
- [x] Task 12: Create `tests/data/test_qc.py` ✓ 2026-04-01
- [x] Task 13: Create `tests/data/test_open_meteo.py` ✓ 2026-04-01

### Validation
- [x] Task 14: Full validation suite (ruff, pyright, pytest) ✓ 2026-04-01

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/config.py` | Create | Pydantic Settings config |
| `src/windcast/data/schema.py` | Create | Canonical SCADA schema |
| `src/windcast/data/kelmarsh.py` | Create | Kelmarsh v4 parser |
| `src/windcast/data/qc.py` | Create | QC pipeline |
| `src/windcast/data/open_meteo.py` | Create | Open-Meteo client |
| `src/windcast/data/__init__.py` | Modify | Add exports |
| `scripts/ingest_kelmarsh.py` | Create | CLI ingestion script |
| `tests/data/__init__.py` | Create | Test sub-package |
| `tests/test_config.py` | Create | Config tests |
| `tests/data/test_schema.py` | Create | Schema tests |
| `tests/data/test_kelmarsh.py` | Create | Parser tests |
| `tests/data/test_qc.py` | Create | QC tests |
| `tests/data/test_open_meteo.py` | Create | Open-Meteo tests |

## Notes

## Completion
- **Started**: 2026-04-01
- **Completed**: 2026-04-01
- **Commit**: (link to commit when done)
