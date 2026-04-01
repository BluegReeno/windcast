# WindCast - Current Status

**Last Updated**: 2026-04-01
**Current Phase**: Phase 1 — Core Pipeline
**Target**: MVP with Kelmarsh data

---

## Current Focus

**Task File**: `.claude/tasks/phase-1-1-project-setup.md` (COMPLETE)

### Priority Order

1. **Phase 1.1 — Project Setup** - DONE ✓
2. **Phase 1.2 — Data Ingestion** - Next up
3. **Phase 1.3 — Feature Engineering** - Pending

---

## What's DONE

### Phase 1.1: Project Setup
- [x] `pyproject.toml` with all deps + tool configs
- [x] `src/windcast/` package structure
- [x] `.gitignore` for Python/data/MLflow
- [x] `uv sync` — 109 packages installed
- [x] Smoke tests passing (3/3)
- [x] Validation: ruff ✓, pyright ✓, pytest ✓

---

## Quick Commands

```bash
# Setup
uv sync

# Validation
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pyright src/
uv run pytest tests/ -v
```

---

## Key Files

```
src/windcast/
├── __init__.py          # Root package
├── data/                # Data ingestion (next)
├── features/            # Feature engineering
└── models/              # ML models
tests/
└── test_setup.py        # Smoke tests
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| Package Manager | uv |
| Data | Polars 1.39 |
| ML | XGBoost 3.2, LightGBM 4.6, scikit-learn 1.8 |
| Tracking | MLflow 3.10 |
| Tuning | Optuna 4.8 |

---

**Next Action**: Start Phase 1.2 — Data Ingestion (Kelmarsh dataset)
